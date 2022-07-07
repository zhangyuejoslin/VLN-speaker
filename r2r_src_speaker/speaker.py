import torch
import numpy as np
from param import args
import os
import utils
import torch.nn.functional as F
import model_PREVALENT
from pytorch_transformers.modeling_bert import BertOnlyMLMHead
from torch import nn

class Speaker():
    env_actions = {
        'left': (0,-1, 0), # left
        'right': (0, 1, 0), # right
        'up': (0, 0, 1), # up
        'down': (0, 0,-1), # down
        'forward': (1, 0, 0), # forward
        '<end>': (0, 0, 0), # <end>
        '<start>': (0, 0, 0), # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, listener, tok):
        self.env = env
        self.feature_size = self.env.feature_size
        self.tok = tok
        self.listener = listener

        # Model
        print("VOCAB_SIZE", self.tok.vocab_size)
        self.speaker_vln_bert = model_PREVALENT.VLNBERT(feature_size=self.feature_size + args.angle_feat_size).cuda()
        self.speaker_vln_bert_optimizer = args.optimizer(self.speaker_vln_bert.parameters(), lr=args.lr)

        # Evaluation
        self.config = self.speaker_vln_bert.vln_bert.config
        self.mlmhead = BertOnlyMLMHead(self.config).cuda()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        
        self.tie_weights()
    
    def tie_weights(self):
        self.speaker_vln_bert.vln_bert._tie_or_clone_weights(self.mlmhead.predictions.decoder, self.speaker_vln_bert.vln_bert.embeddings.word_embeddings)

    def train(self, iters):
        for i in range(iters):
            self.env.reset()

            self.speaker_vln_bert_optimizer.zero_grad()

            loss = self.teacher_forcing(train=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm(self.speaker_vln_bert_optimizer.parameters(), 40.)
            self.peaker_vln_bert_optimizer.step()

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True)
        path2inst = {}
        total = self.env.size()
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch()  # Get the insts of the result
            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = self.shrink_bert(inst)  # Shrink the words
        return path2inst
    
    def shrink_bert(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <SEP> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.tok.vocab['[SEP]'])     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.tok.vocab['[CLS]']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]
    
    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.tok.vocab['[PAD]']:
                break
            else:
                sentence.append(self.tok.ids_to_tokens[ix])
        return " ".join(sentence)

    def valid(self, *aargs, **kwargs):
        """

        :param iters:
        :return: path2inst: path_id --> inst (the number from <bos> to <eos>)
                 loss: The XE loss
                 word_accu: per word accuracy
                 sent_accu: per sent accuracy
        """
        path2inst = self.get_insts(*aargs, **kwargs)

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 1 if args.fast_train else 3     # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False))
        metrics /= N

        return (path2inst, *metrics)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12   # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                    # print("UP")
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                    # print("DOWN")
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                    # print("RIGHT")
                    # print(self.env.env.sims[idx].getState().viewIndex, trg_point)
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions):
        candidate_feat = np.zeros((len(obs), self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['feature'] # Image feat
        return torch.from_numpy(candidate_feat).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        """
        :param viewpoints: [[], [], ....(batch_size)]. Only for dropout viewpoint
        :param get_first_feat: whether output the first feat
        :return:
        """
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []
        first_feat = np.zeros((len(obs), self.feature_size+args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            first_feat[i, -args.angle_feat_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
        first_feat = torch.from_numpy(first_feat).cuda()
        while not ended.all():
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self.listener._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended)
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action
            can_feats.append(self._candidate_variable(obs, teacher_action))
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1).contiguous()  # batch_size, max_len, 36, 2052
        can_feats = torch.stack(can_feats, 1).contiguous()  # batch_size, max_len, 2052
        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats), length

    def gt_words(self, obs):
        """
        See "utils.Tokenizer.encode_sentence(...)" for "instr_encoding" details
        """
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        return torch.from_numpy(seq_tensor).cuda()

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False):
        if train:
            self.speaker_vln_bert.train()
        else:
            self.speaker_vln_bert.eval()

        ### Obtain the instructions and image features
    
        obs = self.env._get_obs()
        batch_size = len(obs)
   
        sentence, language_attention_mask, token_type_ids, \
            seq_lengths, perm_idx = self.listener._sort_batch(obs)

        ### TODO: mask tokens

        ''' Language BERT '''
        language_inputs = {'mode':        'language',
                        'sentence':       sentence,
                        'attention_mask': language_attention_mask,
                        'lang_mask':      language_attention_mask,
                        'token_type_ids': token_type_ids}
        h_t, language_features = self.speaker_vln_bert(**language_inputs)

        ###
        (pano_img_feats, candidate_feat), candidate_leng = self.from_shortest_path()

        ## visual mask
        visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
        visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

        visual_inputs = {'mode':              'speaker',
                        'sentence':           h_t,
                        'attention_mask':     visual_attention_mask,
                        'lang_mask':          language_attention_mask,
                        'vis_mask':           visual_temp_mask,
                        'token_type_ids':     token_type_ids,
                        'cand_feats':         candidate_feat,
                        }
        h_t, logits  = self.speaker_vln_bert(**visual_inputs)
        prediction_scores = self.mlmhead(logits)
        loss = self.criterion(prediction_scores.view(-1, self.config.vocab_size), sentence.view(-1))

        if train:
            return loss
        
        # else:
        #     # Evaluation
        #     _, predict = logits.max(dim=1)                                  # BATCH, LENGTH
        #     gt_mask = (insts != self.tok.vocab['[PAD]'])
        #     correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
        #     correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
        #     word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
        #     sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
        #     return loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                     log_probs(torch, requires_grad, [batch,max_len])
                                     hiddens(torch, requires_grad, [batch, max_len, dim})
                      And if train: the log_probs and hiddens are detached
                 if not sampling: returns insts(np, [batch, max_len])
        """
        if train:
            self.speaker_vln_bert.train()
        else:
            self.speaker_vln_bert.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        
        # # This code block is only used for the featdrop.
        # if featdropmask is not None:
        #     img_feats[..., :-args.angle_feat_size] *= featdropmask
        #     candidate_feat[..., :-args.angle_feat_size] *= featdropmask
        
        ended = np.zeros(len(obs), np.bool)
        word = np.ones(len(obs), np.int64) * self.tok.vocab['[CLS]'] 
        word = torch.from_numpy(word).view(-1, 1).cuda()

        ### TODO to create work mask and tokens
        language_attention_mask = torch.ones(len(obs), 1).cuda()
        token_type_ids = torch.zeros_like(language_attention_mask).cuda()

        
        viewpoints_list = [list() for _ in range(batch_size)]
        (img_feats, candidate_feat), candidate_leng = self.from_shortest_path(viewpoints=viewpoints_list)      # Image Feature (from the shortest path)
        visual_temp_mask = (utils.length2mask(candidate_leng) == 0).long()
        visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1)

        words = []
        log_probs = []
        hidden_states = []
        entropies = []

        for steps in range(args.maxDecode):    
            language_inputs = {'mode': 'language',
                           'sentence': word,
                           'attention_mask': language_attention_mask,
                           'lang_mask':      language_attention_mask,
                           'token_type_ids': token_type_ids}
            _, language_features = self.speaker_vln_bert(**language_inputs)

            h_t = language_features
            # Decode Step
            visual_inputs = {'mode':          'speaker',
                        'sentence':           language_features,
                        'attention_mask':     visual_attention_mask,
                        'lang_mask':          language_attention_mask,
                        'vis_mask':           visual_temp_mask,
                        'cand_feats':         candidate_feat,
                        }
            h_t, logits  = self.speaker_vln_bert(**visual_inputs)
            h_t = h_t.unsqueeze(1)

            # Select the word
            logits = logits.squeeze()                                           # logits: (b, vocab_size)
            logits[:, self.tok.vocab['[UNK]']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    hidden_states.append(h_t.squeeze())
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    hidden_states.append(h_t.squeeze().detach())
                    entropies.append(m.entropy().detach())
            else:
                values, word = logits.max(1)

            # Append the word
            cpu_word = word.cpu().numpy()
            cpu_word[ended] = self.tok.vocab['[PAD]']
            words.append(cpu_word)

            # Prepare the shape for next step
            word = word.view(-1, 1)

            # End?
            ended = np.logical_or(ended, cpu_word == self.tok.vocab['[SEP]'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(hidden_states, 1), torch.stack(entropies, 1)
        else:
            return np.stack(words, 1)     # [(b), (b), (b), ...] --> [b, l]

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            # print(name)
            # print(list(model.state_dict().keys()))
            # for key in list(model.state_dict().keys()):
            #     print(key, model.state_dict()[key].size())
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1
