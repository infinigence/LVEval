# Error cases

In the hotpotwikiqa_mixup_16k dataset, In the 65th test sample, the question is "What is the date of death of the director of the film Nallavan Vazhvan?" This question requires multi-hop reasoning to get the correct answer. The model needs to first extract the director's name from the passage "### Passage 30" which introduces the film Nallavan Vazhvan, and then retrieve the final answer from "### Passage 15" which details the director's life. But the model gives the wrong answer from another director's introduction in "### Passage 27", which provides a vague statement—that the director was involved in the production of over 60 films—which the model mistakenly interprets as relevant information for answering the question. Another potential reason for the model's incorrect response is that the director's first name in the film's introduction was abbreviated, which prevented the model from retrieving the correct answer through exact matching. This indicates that even a powerful model like Llama 3 struggles to accurately understand the relationships between entities in long-context multi-step reasoning and is easily misled by other seemingly straightforward yet ambiguous information.

```json
{"pred": "9 December 1988", "answers": ["4 November 2003"], "gold_ans": "4 November 2003", "input": "What is the date of death of the director of film Nallavan Vazhvan?", "all_classes": null, "length": 21447}
```

Context: 

```json
"......### Passage 15\nPalaniyaandi Neelakantan (2 October 1916 \u2013 4 November 2003) was a Tamil film director, who was active for nearly four decades.\n\nLife\nHe was born at Villupuram, Tamil Nadu......### Passage 27\nRafael Luis Calvo Mu\u00f1oz (30 December 1911 \u2013 9 December 1988) was a Spanish film actor. He appeared in more than 60 films including Miracle of Marcelino (1955)....### Passage 30\nNallavan Vazhvan (transl.\u2009The good man will live) is a 1961 Indian Tamil-language crime thriller film produced and directed by P. Neelakantan......"
```
In the factrecall-zh-16k dataset, all of Llama-3-8b-Instruct's responses were misled by the CF, that is "贝克汉姆"( "David Beckham"), whereas in the factrecall-en-16k dataset, Llama-3-8b-Instruct has only 32% of samples being misled by the CF. This suggests that the model's anti-interference ability may be highly imbalanced across different languages. We provide a list of model responses as examples for clarification.

factrecall-zh-16k
```json
# factrecall-zh-16k
{"pred": "贝克汉姆。", "answers": ["贝多芬"], "gold_ans": null, "input": "被世人广泛推崇为现代物理学奠基人的科学家叫什么名字？", "all_classes": null, "length": 13249}
{"pred": "贝克汉姆。", "answers": ["贝多芬"], "gold_ans": null, "input": "被世人广泛推崇为现代物理学奠基人的科学家叫什么名字？", "all_classes": null, "length": 13390}
{"pred": "贝克汉姆", "answers": ["贝多芬"], "gold_ans": null, "input": "被世人广泛推崇为现代物理学奠基人的科学家叫什么名字？", "all_classes": null, "length": 13316}
{"pred": "贝克汉姆。", "answers": ["贝多芬"], "gold_ans": null, "input": "被世人广泛推崇为现代物理学奠基人的科学家叫什么名字？", "all_classes": null, "length": 13334}
{"pred": "贝克汉姆。", "answers": ["贝多芬"], "gold_ans": null, "input": "被世人广泛推崇为现代物理学奠基人的科学家叫什么名字？", "all_classes": null, "length": 13266}
......
```
factrecall-en-16k
```json
# factrecall-en-16k
{"pred": "David Beckham.", "answers": ["Ludwig Beethoven"], "gold_ans": null, "input": "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?", "all_classes": null, "length": 13940}
{"pred": "David Beckham.", "answers": ["Ludwig Beethoven"], "gold_ans": null, "input": "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?", "all_classes": null, "length": 14047}
{"pred": "Ludwig Beethoven.", "answers": ["Ludwig Beethoven"], "gold_ans": null, "input": "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?", "all_classes": null, "length": 13988}
{"pred": "Ludwig Beethoven.", "answers": ["Ludwig Beethoven"], "gold_ans": null, "input": "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?", "all_classes": null, "length": 14126}
{"pred": "Ludwig Beethoven.", "answers": ["Ludwig Beethoven"], "gold_ans": null, "input": "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?", "all_classes": null, "length": 13895}
{"pred": "Ludwig Beethoven.", "answers": ["Ludwig Beethoven"], "gold_ans": null, "input": "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?", "all_classes": null, "length": 14019}
......
```

# Metric cases
For AK related falsely high score in plain F1, please look at the case below. The model failed to locate the specific time information (2020), but it gets a 0.3 score due to other matched words like "independent publishing of digital books". In human evaluation, this response would clearly be given a score of 0. These types of examples have been manually verified as a common occurrence across multiple datasets in plain F1 metric, so we designed the keyword-recall-based metric to reduce bias.

```json
{"qa_f1_score": 0.30769230769230765, 
"pred": "There is no mention of Martin or independent publishing of digital books in the passage. The passage appears to be about a research paper on contour completion using deep structure priors.", 
"answers": ["Martin began independent publishing her books as digital books in **2020**."], 
"answers_keywords": "2020", 
"input": "When did Martin start independent publishing her books as digital books?", "all_classes": null, "length": 18496}
```

More examples：

```json
{"qa_f1_score_with_gold_ans": 0.4, 
"pred": "For services to Medicine and to the community in the Cayman Islands.", 
"answers": ["For his services to **music**."], 
"answers_keyword": "services to music", 
"input": "What is Geoffrey Michael Windsor Taylor being recognized for?", "all_classes": null, "length": 32957}
```
```json
{"qa_f1_score": 0.6666666666666666, 
"pred": "Low mechanical flexibility.", 
"answers": ["**Increased** mechanical flexibility."], 
"answers_keyword": "Increased mechanical flexibility.", 
"input": "What are the benefits of using binary variables in the SLAS formulation?", "all_classes": null, "length": 16690}
```
```json
{"qa_f1_zh_score": 0.4827586206896552, 
"pred": "根据文章26中的内容，电影《毕业风暴》的导演是提莫·贝克曼贝托夫（TimurBekmambetov）。", 
"answers": ["《毕业风暴》的导演是罗马尼亚导演**克里斯汀穆基**。"], 
"answers_keyword": "克里斯汀穆基", 
"input": "谁是电影《毕业风暴》的导演？", "all_classes": null, "length": 16114}
```
