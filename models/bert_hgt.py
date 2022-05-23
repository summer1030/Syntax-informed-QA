
"""
*******************************************************************

Models

Including the following models,

* BERT_HGT_CON: Build constituency graph on top of backbone
* BERT_HGT_DEP: Build dependency graph on top of backbone
* BERT_HGT_CON_AND_DEP: Build constituency and dependency graph on top of backbone

*******************************************************************
"""


import torch_scatter
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from custom_squad_methods import pad_sequence
from transformers import AutoTokenizer, AutoModel, BertModel, BertPreTrainedModel, BertForQuestionAnswering
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, GATConv
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import numpy as np

from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, FastRGCNConv, GATConv, HGTConv, Linear

import pickle
import joblib


# dependency relationships for reference
dep_type_mapping={'type0': 'conj', 'type1': 'csubjpass', 'type2': 'prep', 'type3': 'cc', 'type4': 'nsubjpass', 'type5': 'advmod', 'type6': 'xcomp', 'type7': 'pcomp', 'type8': 'dobj', 'type9': 'preconj', 'type10': 'advcl', 'type11': 'nsubj', 'type12': 'poss', 'type13': 'npadvmod', 'type14': 'dep', 'type15': 'rcmod', 'type16': 'parataxis', 'type17': 'infmod', 'type18': 'number', 'type19': 'partmod', 'type20': 'discourse', 'type21': 'pobj', 'type22': 'neg', 'type23': 'quantmod', 'type24': 'prt', 'type25': 'ccomp', 'type26': 'appos', 'type27': 'iobj', 'type28': 'cop', 'type29': 'nn', 'type30': 'num', 'type31': 'amod', 'type32': 'aux', 'type33': 'mwe', 'type34': 'mark', 'type35': 'tmod', 'type36': 'det', 'type37': 'auxpass', 'type38': 'possessive', 'type39': 'acomp', 'type40': 'csubj', 'type41': 'punct', 'type42': 'expl', 'type43': 'predet'}

# entity relationships for reference
ent_type_mapping={'org:alternate_names': 0, 'org:city_of_headquarters': 1, 'org:country_of_headquarters': 2, 'org:dissolved': 3, 'org:founded': 4, 'org:founded_by': 5, 'org:member_of': 6, 'org:members': 7, 'org:number_of_employees/members': 8, 'org:parents': 9, 'org:political/religious_affiliation': 10, 'org:shareholders': 11, 'org:stateorprovince_of_headquarters': 12, 'org:subsidiaries': 13, 'org:top_members/employees': 14, 'org:website': 15, 'per:age': 16, 'per:alternate_names': 17, 'per:cause_of_death': 18, 'per:charges': 19, 'per:children': 20, 'per:cities_of_residence': 21, 'per:city_of_birth': 22, 'per:city_of_death': 23, 'per:countries_of_residence': 24, 'per:country_of_birth': 25, 'per:country_of_death': 26, 'per:date_of_birth': 27, 'per:date_of_death': 28, 'per:employee_of': 29, 'per:origin': 30, 'per:other_family': 31, 'per:parents': 32, 'per:religion': 33, 'per:schools_attended': 34, 'per:siblings': 35, 'per:spouse': 36, 'per:stateorprovince_of_birth': 37, 'per:stateorprovince_of_death': 38, 'per:stateorprovinces_of_residence': 39, 'per:title': 40, 'subword2word':41, 'word2phrase':42}


class BERT_HGT_CON(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self,
                config,
                num_dims,
                num_edge_types,
                num_hgt_layers = 2,
                num_hgt_heads = 4,
                graph_data_root = "./data/squad_files/constituency_graphs",
                ):
        super().__init__(config)
        self.num_labels = config.num_labels
        print("num dims {}".format(num_dims))
        self.graph_data_root = graph_data_root

        # build the backbone
        self.backbone = BertModel(config, add_pooling_layer=False)

        # initialize a heterogeneous graph
        graph_data = HeteroData()
        graph_data['token'].x = None
        graph_data['virtual'].x = None
        graph_data['token','connects','token'].edge_index = None
        graph_data['token','belongs','virtual'].edge_index = None
        graph_data['virtual','consists','virtual'].edge_index = None

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph_data.node_types:
            self.lin_dict[node_type] = Linear(-1, config.hidden_size)

        self.num_hgt_layers = num_hgt_layers
        self.num_heads = num_hgt_heads
        self.hgt_convs = torch.nn.ModuleList()
        for _ in range(self.num_hgt_layers):
            conv = HGTConv(config.hidden_size, config.hidden_size, graph_data.metadata(),
                           self.num_heads, group='sum')
            self.hgt_convs.append(conv)

        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_virtual_node_embeds(self, qas_ids):
        virtual_node_embeds = []
        for each in qas_ids:
            virtual_node_embeds += [pickle.load(open('{}/{}'.format(self.graph_data_root,each),'rb'))['virtual_node_embeds']]
        return torch.cat(virtual_node_embeds,dim=0)

    def add_word_token_embeds(self, bert_embeds, mapping):
        added_node_embeds = []
        for j in range(mapping.size()[1]):
            start_token_pos, end_token_pos = mapping[1,j],mapping[2,j]
            added_node_embeds += [torch.mean(bert_embeds[start_token_pos:end_token_pos],dim=0)]
        added_node_embeds = torch.stack(added_node_embeds)

        return torch.cat((bert_embeds,added_node_embeds),dim=0)

    def recover_logits(self, logits, mapping, sent_start_index):
        updated_logits = []
        start_pos = 0
        count=0
        while count<len(sent_start_index):
            end_pos=start_pos+384
            updated_logits += [logits[start_pos:end_pos,:].unsqueeze(0)]
            if mapping[0,sent_start_index[count]] == 0:
                start_pos = end_pos
            else:
                try:
                    start_pos = end_pos+sent_start_index[count+1]-sent_start_index[count]
                except:
                    pass
            count+=1

        try:
            return torch.cat(updated_logits,dim=0)
        except:
            import pdb;pdb.set_trace()

    def forward(self, data,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids = data.input_ids
        attention_mask = data.attention_mask
        token_type_ids = data.token_type_ids
        num_nodes = data.num_nodes

        if hasattr(data, 'start_position'):
            start_positions = data.start_position
            end_positions = data.end_position
        else:
            start_positions = None
            end_positions = None

        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # get the outpus on the backbone
        sequence_output = outputs[0]

        # begin to build the graph
        graph_data = HeteroData()

        sent_start_index=(data.added_node_mapping[0,:]==384).nonzero()
        passed_sent_index=(data.added_node_mapping[0,:]==0).nonzero()
        all_sent_start_index = sorted(torch.cat((sent_start_index,passed_sent_index),0))

        # flatten all data
        initial_reps_list = []
        for i, sample_output in enumerate(sequence_output):
            if data.added_node_mapping[0,all_sent_start_index[i]] != 0:
                if i<= len(all_sent_start_index)-2:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:all_sent_start_index[i+1]])
                    initial_reps_list.append(tmp)
                elif i == len(all_sent_start_index)-1:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:])
                    initial_reps_list.append(tmp)
            else:
                initial_reps_list.append(sample_output)

        initial_reps = torch.cat(initial_reps_list)

        graph_data['token'].x = initial_reps
        graph_data['virtual'].x = self.get_virtual_node_embeds(data.qas_id).to(input_ids.device)
        graph_data['token','connects','token'].edge_index = data.t2t_edge_index
        graph_data['token','belongs','virtual'].edge_index = data.t2v_edge_index
        graph_data['virtual','consists','virtual'].edge_index = data.v2v_edge_index

        self.lin_dict.cuda()

        for node_type, x in graph_data.x_dict.items():
            graph_data.x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.hgt_convs:
            graph_data.x_dict = conv(graph_data.x_dict, graph_data.edge_index_dict)

        # fed the 'token' nodes into output layers to generate logits
        logits = self.qa_outputs(graph_data.x_dict['token'])

        # recover the logits to the original size
        updated_logits = self.recover_logits(logits, data.added_node_mapping, all_sent_start_index)

        start_logits, end_logits = updated_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BERT_HGT_DEP(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config,
                    num_dims,
                    num_edge_types,
                    num_hgt_layers = 2,
                    num_hgt_heads = 4
                    ):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)

        # initialize a heterogeneous graph
        graph_data = HeteroData()
        graph_data['token'].x = None
        graph_data['token','consists','token'].edge_index = None
        graph_data['token','type0','token'].edge_index = None
        graph_data['token','type1','token'].edge_index = None
        graph_data['token','type2','token'].edge_index = None
        graph_data['token','type3','token'].edge_index = None
        graph_data['token','type4','token'].edge_index = None
        graph_data['token','type5','token'].edge_index = None
        graph_data['token','type6','token'].edge_index = None
        graph_data['token','type7','token'].edge_index = None
        graph_data['token','type8','token'].edge_index = None
        graph_data['token','type9','token'].edge_index = None
        graph_data['token','type10','token'].edge_index = None
        graph_data['token','type11','token'].edge_index = None
        graph_data['token','type12','token'].edge_index = None
        graph_data['token','type13','token'].edge_index = None
        graph_data['token','type14','token'].edge_index = None
        graph_data['token','type15','token'].edge_index = None
        graph_data['token','type16','token'].edge_index = None
        graph_data['token','type17','token'].edge_index = None
        graph_data['token','type18','token'].edge_index = None
        graph_data['token','type19','token'].edge_index = None
        graph_data['token','type20','token'].edge_index = None
        graph_data['token','type21','token'].edge_index = None
        graph_data['token','type22','token'].edge_index = None
        graph_data['token','type23','token'].edge_index = None
        graph_data['token','type24','token'].edge_index = None
        graph_data['token','type25','token'].edge_index = None
        graph_data['token','type26','token'].edge_index = None
        graph_data['token','type27','token'].edge_index = None
        graph_data['token','type28','token'].edge_index = None
        graph_data['token','type29','token'].edge_index = None
        graph_data['token','type30','token'].edge_index = None
        graph_data['token','type31','token'].edge_index = None
        graph_data['token','type32','token'].edge_index = None
        graph_data['token','type33','token'].edge_index = None
        graph_data['token','type34','token'].edge_index = None
        graph_data['token','type35','token'].edge_index = None
        graph_data['token','type36','token'].edge_index = None
        graph_data['token','type37','token'].edge_index = None
        graph_data['token','type38','token'].edge_index = None
        graph_data['token','type39','token'].edge_index = None
        graph_data['token','type40','token'].edge_index = None
        graph_data['token','type41','token'].edge_index = None
        graph_data['token','type42','token'].edge_index = None
        graph_data['token','type43','token'].edge_index = None

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph_data.node_types:
            self.lin_dict[node_type] = Linear(-1, config.hidden_size)

        self.num_hgt_layers = num_hgt_layers
        self.num_heads = num_hgt_heads
        self.hgt_convs = torch.nn.ModuleList()
        for _ in range(self.num_hgt_layers):
            conv = HGTConv(config.hidden_size, config.hidden_size, graph_data.metadata(),
                           self.num_heads, group='sum')
            self.hgt_convs.append(conv)

        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def add_word_token_embeds(self, bert_embeds, mapping):
        added_node_embeds = []
        for j in range(mapping.size()[1]):
            start_token_pos, end_token_pos = mapping[1,j],mapping[2,j]
            added_node_embeds += [torch.mean(bert_embeds[start_token_pos:end_token_pos],dim=0)]
        added_node_embeds = torch.stack(added_node_embeds)

        return torch.cat((bert_embeds,added_node_embeds),dim=0)

    def recover_logits(self, logits, mapping, sent_start_index):
        updated_logits = []
        start_pos = 0

        count=0
        while count<len(sent_start_index):
            end_pos=start_pos+384
            updated_logits += [logits[start_pos:end_pos,:].unsqueeze(0)]
            if mapping[0,sent_start_index[count]] == 0:
                start_pos = end_pos
            else:
                try:
                    start_pos = end_pos+sent_start_index[count+1]-sent_start_index[count]
                except:
                    pass
            count+=1

        return torch.cat(updated_logits,dim=0)


    def forward(self, data,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids = data.input_ids
        attention_mask = data.attention_mask
        token_type_ids = data.token_type_ids

        num_nodes = data.num_nodes

        if hasattr(data, 'start_position'):
            start_positions = data.start_position
            end_positions = data.end_position
        else:
            start_positions = None
            end_positions = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # initialize a heterogeneous graph
        graph_data = HeteroData()

        sent_start_index=(data.added_node_mapping[0,:]==384).nonzero()
        passed_sent_index=(data.added_node_mapping[0,:]==0).nonzero()
        all_sent_start_index = sorted(torch.cat((sent_start_index,passed_sent_index),0))

        initial_reps_list = []
        for i, sample_output in enumerate(sequence_output):
            if data.added_node_mapping[0,all_sent_start_index[i]] != 0:
                if i<= len(all_sent_start_index)-2:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:all_sent_start_index[i+1]])
                    initial_reps_list.append(tmp)
                elif i == len(all_sent_start_index)-1:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:])
                    initial_reps_list.append(tmp)
            else:
                initial_reps_list.append(sample_output)

        initial_reps = torch.cat(initial_reps_list)

        graph_data['token'].x = initial_reps
        graph_data['token','consists','token'].edge_index = data.subword2word_edge_index
        graph_data['token','type0','token'].edge_index = data.type0_edge_index
        graph_data['token','type1','token'].edge_index = data.type1_edge_index
        graph_data['token','type2','token'].edge_index = data.type2_edge_index
        graph_data['token','type3','token'].edge_index = data.type3_edge_index
        graph_data['token','type4','token'].edge_index = data.type4_edge_index
        graph_data['token','type5','token'].edge_index = data.type5_edge_index
        graph_data['token','type6','token'].edge_index = data.type6_edge_index
        graph_data['token','type7','token'].edge_index = data.type7_edge_index
        graph_data['token','type8','token'].edge_index = data.type8_edge_index
        graph_data['token','type9','token'].edge_index = data.type9_edge_index
        graph_data['token','type10','token'].edge_index = data.type10_edge_index
        graph_data['token','type11','token'].edge_index = data.type11_edge_index
        graph_data['token','type12','token'].edge_index = data.type12_edge_index
        graph_data['token','type13','token'].edge_index = data.type13_edge_index
        graph_data['token','type14','token'].edge_index = data.type14_edge_index
        graph_data['token','type15','token'].edge_index = data.type15_edge_index
        graph_data['token','type16','token'].edge_index = data.type16_edge_index
        graph_data['token','type17','token'].edge_index = data.type17_edge_index
        graph_data['token','type18','token'].edge_index = data.type18_edge_index
        graph_data['token','type19','token'].edge_index = data.type19_edge_index
        graph_data['token','type20','token'].edge_index = data.type20_edge_index
        graph_data['token','type21','token'].edge_index = data.type21_edge_index
        graph_data['token','type22','token'].edge_index = data.type22_edge_index
        graph_data['token','type23','token'].edge_index = data.type23_edge_index
        graph_data['token','type24','token'].edge_index = data.type24_edge_index
        graph_data['token','type25','token'].edge_index = data.type25_edge_index
        graph_data['token','type26','token'].edge_index = data.type26_edge_index
        graph_data['token','type27','token'].edge_index = data.type27_edge_index
        graph_data['token','type28','token'].edge_index = data.type28_edge_index
        graph_data['token','type29','token'].edge_index = data.type29_edge_index
        graph_data['token','type30','token'].edge_index = data.type30_edge_index
        graph_data['token','type31','token'].edge_index = data.type31_edge_index
        graph_data['token','type32','token'].edge_index = data.type32_edge_index
        graph_data['token','type33','token'].edge_index = data.type33_edge_index
        graph_data['token','type34','token'].edge_index = data.type34_edge_index
        graph_data['token','type35','token'].edge_index = data.type35_edge_index
        graph_data['token','type36','token'].edge_index = data.type36_edge_index
        graph_data['token','type37','token'].edge_index = data.type37_edge_index
        graph_data['token','type38','token'].edge_index = data.type38_edge_index
        graph_data['token','type39','token'].edge_index = data.type39_edge_index
        graph_data['token','type40','token'].edge_index = data.type40_edge_index
        graph_data['token','type41','token'].edge_index = data.type41_edge_index
        graph_data['token','type42','token'].edge_index = data.type42_edge_index
        graph_data['token','type43','token'].edge_index = data.type43_edge_index


        self.lin_dict.cuda()

        for node_type, x in graph_data.x_dict.items():
            graph_data.x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.hgt_convs:
            graph_data.x_dict = conv(graph_data.x_dict, graph_data.edge_index_dict)

        logits = self.qa_outputs(graph_data.x_dict['token'])

        updated_logits = self.recover_logits(logits, data.added_node_mapping, all_sent_start_index)


        start_logits, end_logits = updated_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BERT_HGT_CON_AND_DEP(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config,
                 num_dims,
                 num_edge_types,
                 num_hgt_layers = 2,
                 num_hgt_heads = 4,
                 graph_data_root = "./data/squad_files/con_and_dep_graphs",
                 ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.graph_data_root = graph_data_root
        self.bert = BertModel(config, add_pooling_layer=False)

        # initialize a heterogeneous graph
        graph_data = HeteroData()
        graph_data['token'].x = None
        graph_data['virtual'].x = None
        graph_data['token','connects','token'].edge_index = None
        graph_data['token','belongs','virtual'].edge_index = None
        graph_data['virtual','consists','virtual'].edge_index = None
        graph_data['token','type0','token'].edge_index = None
        graph_data['token','type1','token'].edge_index = None
        graph_data['token','type2','token'].edge_index = None
        graph_data['token','type3','token'].edge_index = None
        graph_data['token','type4','token'].edge_index = None
        graph_data['token','type5','token'].edge_index = None
        graph_data['token','type6','token'].edge_index = None
        graph_data['token','type7','token'].edge_index = None
        graph_data['token','type8','token'].edge_index = None
        graph_data['token','type9','token'].edge_index = None
        graph_data['token','type10','token'].edge_index = None
        graph_data['token','type11','token'].edge_index = None
        graph_data['token','type12','token'].edge_index = None
        graph_data['token','type13','token'].edge_index = None
        graph_data['token','type14','token'].edge_index = None
        graph_data['token','type15','token'].edge_index = None
        graph_data['token','type16','token'].edge_index = None
        graph_data['token','type17','token'].edge_index = None
        graph_data['token','type18','token'].edge_index = None
        graph_data['token','type19','token'].edge_index = None
        graph_data['token','type20','token'].edge_index = None
        graph_data['token','type21','token'].edge_index = None
        graph_data['token','type22','token'].edge_index = None
        graph_data['token','type23','token'].edge_index = None
        graph_data['token','type24','token'].edge_index = None
        graph_data['token','type25','token'].edge_index = None
        graph_data['token','type26','token'].edge_index = None
        graph_data['token','type27','token'].edge_index = None
        graph_data['token','type28','token'].edge_index = None
        graph_data['token','type29','token'].edge_index = None
        graph_data['token','type30','token'].edge_index = None
        graph_data['token','type31','token'].edge_index = None
        graph_data['token','type32','token'].edge_index = None
        graph_data['token','type33','token'].edge_index = None
        graph_data['token','type34','token'].edge_index = None
        graph_data['token','type35','token'].edge_index = None
        graph_data['token','type36','token'].edge_index = None
        graph_data['token','type37','token'].edge_index = None
        graph_data['token','type38','token'].edge_index = None
        graph_data['token','type39','token'].edge_index = None
        graph_data['token','type40','token'].edge_index = None
        graph_data['token','type41','token'].edge_index = None
        graph_data['token','type42','token'].edge_index = None
        graph_data['token','type43','token'].edge_index = None

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph_data.node_types:
            self.lin_dict[node_type] = Linear(-1, config.hidden_size)

        self.num_hgt_layers = num_hgt_layers
        self.num_heads = num_hgt_heads
        self.hgt_convs = torch.nn.ModuleList()
        for _ in range(self.num_hgt_layers):
            conv = HGTConv(config.hidden_size, config.hidden_size, graph_data.metadata(),
                           self.num_heads, group='sum')
            self.hgt_convs.append(conv)

        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_virtual_node_embeds(self, qas_ids):
        virtual_node_embeds = []
        for each in qas_ids:
            virtual_node_embeds += [pickle.load(open('{}/{}'.format(self.graph_data_root,each),'rb'))['virtual_node_embeds']]
        return torch.cat(virtual_node_embeds,dim=0)

    def add_word_token_embeds(self, bert_embeds, mapping):
        added_node_embeds = []
        for j in range(mapping.size()[1]):
            start_token_pos, end_token_pos = mapping[1,j],mapping[2,j]
            added_node_embeds += [torch.mean(bert_embeds[start_token_pos:end_token_pos],dim=0)]
        added_node_embeds = torch.stack(added_node_embeds)

        return torch.cat((bert_embeds,added_node_embeds),dim=0)

    def recover_logits(self, logits, mapping, sent_start_index):
        updated_logits = []
        start_pos = 0

        count=0
        while count<len(sent_start_index):
            end_pos=start_pos+384
            updated_logits += [logits[start_pos:end_pos,:].unsqueeze(0)]
            if mapping[0,sent_start_index[count]] == 0:
                start_pos = end_pos
            else:
                try:
                    start_pos = end_pos+sent_start_index[count+1]-sent_start_index[count]
                except:
                    pass
            count+=1

        return torch.cat(updated_logits,dim=0)


    def forward(self, data,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids = data.input_ids
        attention_mask = data.attention_mask
        token_type_ids = data.token_type_ids

        num_nodes = data.num_nodes

        if hasattr(data, 'start_position'):
            start_positions = data.start_position
            end_positions = data.end_position
        else:
            start_positions = None
            end_positions = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        graph_data = HeteroData()

        """
        initial_reps_list = []
        for i, sample_output in enumerate(sequence_output):
            initial_reps_list.append(sample_output)
        initial_reps = torch.cat(initial_reps_list)
        graph_data['token'].x = initial_reps
        graph_data['token','connects','token'].edge_index = edge_index
        """

        sent_start_index=(data.added_node_mapping[0,:]==384).nonzero()
        passed_sent_index=(data.added_node_mapping[0,:]==0).nonzero()
        all_sent_start_index = sorted(torch.cat((sent_start_index,passed_sent_index),0))

        initial_reps_list = []
        for i, sample_output in enumerate(sequence_output):
            if data.added_node_mapping[0,all_sent_start_index[i]] != 0:
                if i<= len(all_sent_start_index)-2:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:all_sent_start_index[i+1]])
                    initial_reps_list.append(tmp)
                elif i == len(all_sent_start_index)-1:
                    tmp=self.add_word_token_embeds(sample_output,data.added_node_mapping[:,all_sent_start_index[i]:])
                    initial_reps_list.append(tmp)
            else:
                initial_reps_list.append(sample_output)

        initial_reps = torch.cat(initial_reps_list)

        graph_data['token'].x = initial_reps
        graph_data['virtual'].x = self.get_virtual_node_embeds(data.qas_id).to(input_ids.device)
        graph_data['token','connects','token'].edge_index = data.t2t_edge_index
        graph_data['token','belongs','virtual'].edge_index = data.t2v_edge_index
        graph_data['virtual','consists','virtual'].edge_index = data.v2v_edge_index
        graph_data['token','type0','token'].edge_index = data.type0_edge_index
        graph_data['token','type1','token'].edge_index = data.type1_edge_index
        graph_data['token','type2','token'].edge_index = data.type2_edge_index
        graph_data['token','type3','token'].edge_index = data.type3_edge_index
        graph_data['token','type4','token'].edge_index = data.type4_edge_index
        graph_data['token','type5','token'].edge_index = data.type5_edge_index
        graph_data['token','type6','token'].edge_index = data.type6_edge_index
        graph_data['token','type7','token'].edge_index = data.type7_edge_index
        graph_data['token','type8','token'].edge_index = data.type8_edge_index
        graph_data['token','type9','token'].edge_index = data.type9_edge_index
        graph_data['token','type10','token'].edge_index = data.type10_edge_index
        graph_data['token','type11','token'].edge_index = data.type11_edge_index
        graph_data['token','type12','token'].edge_index = data.type12_edge_index
        graph_data['token','type13','token'].edge_index = data.type13_edge_index
        graph_data['token','type14','token'].edge_index = data.type14_edge_index
        graph_data['token','type15','token'].edge_index = data.type15_edge_index
        graph_data['token','type16','token'].edge_index = data.type16_edge_index
        graph_data['token','type17','token'].edge_index = data.type17_edge_index
        graph_data['token','type18','token'].edge_index = data.type18_edge_index
        graph_data['token','type19','token'].edge_index = data.type19_edge_index
        graph_data['token','type20','token'].edge_index = data.type20_edge_index
        graph_data['token','type21','token'].edge_index = data.type21_edge_index
        graph_data['token','type22','token'].edge_index = data.type22_edge_index
        graph_data['token','type23','token'].edge_index = data.type23_edge_index
        graph_data['token','type24','token'].edge_index = data.type24_edge_index
        graph_data['token','type25','token'].edge_index = data.type25_edge_index
        graph_data['token','type26','token'].edge_index = data.type26_edge_index
        graph_data['token','type27','token'].edge_index = data.type27_edge_index
        graph_data['token','type28','token'].edge_index = data.type28_edge_index
        graph_data['token','type29','token'].edge_index = data.type29_edge_index
        graph_data['token','type30','token'].edge_index = data.type30_edge_index
        graph_data['token','type31','token'].edge_index = data.type31_edge_index
        graph_data['token','type32','token'].edge_index = data.type32_edge_index
        graph_data['token','type33','token'].edge_index = data.type33_edge_index
        graph_data['token','type34','token'].edge_index = data.type34_edge_index
        graph_data['token','type35','token'].edge_index = data.type35_edge_index
        graph_data['token','type36','token'].edge_index = data.type36_edge_index
        graph_data['token','type37','token'].edge_index = data.type37_edge_index
        graph_data['token','type38','token'].edge_index = data.type38_edge_index
        graph_data['token','type39','token'].edge_index = data.type39_edge_index
        graph_data['token','type40','token'].edge_index = data.type40_edge_index
        graph_data['token','type41','token'].edge_index = data.type41_edge_index
        graph_data['token','type42','token'].edge_index = data.type42_edge_index
        graph_data['token','type43','token'].edge_index = data.type43_edge_index

        self.lin_dict.cuda()

        for node_type, x in graph_data.x_dict.items():
            graph_data.x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.hgt_convs:
            graph_data.x_dict = conv(graph_data.x_dict, graph_data.edge_index_dict)
        logits = self.qa_outputs(graph_data.x_dict['token'])

        updated_logits = self.recover_logits(logits, data.added_node_mapping, all_sent_start_index)


        start_logits, end_logits = updated_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
