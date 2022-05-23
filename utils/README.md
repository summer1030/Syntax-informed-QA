Please download the intermediate data files [here](https://drive.google.com/file/d/1koPrjS7zfrX74YLdysE_9Jyxh9VcttLh/view?usp=sharing), and uncompress them to the current directory.

```
- all_paragraphs_dev.json: intermediate file
- all_paragraphs_train.json: intermediate file
- con_parsed_dev.json: constituency parsed results for development set
- con_parsed_train.json: constituency parsed results for training set
- dep_parsed_dev.json: dependency parsed results for development set
- dep_parsed_dev.json: dependency parsed results for training set
```

tag_embeds_dict: The key is type of constituent, e.g., NP (noun phrase) or VP (verb phrase). The value is the intialized tensor for each type of constituent.

dep_rels_tag_list.txt: show all dependency relationships 
