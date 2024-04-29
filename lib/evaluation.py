"""
TODO: Evaluation for a RAG system
question -> *retriever* -> context -> *generator* -> answer

For each text node: question, context, answer
We need to sanitize them for safty, hallucination, etc.
- functions to rewrite the text: t = f_rewrite(t)
  - need to be able to register new rewrite functions
- functions to check the text: f_check(t) -> [float, Explanation]
We can generalize this to a NodeEval class

For each edge between nodes: retriever, generator
We need to evaluate the edge for correctness, etc.
- functions to check the edge
  - For retriever: f_context_relevance(q, c) -> [float, Explanation]
  - For generator: f_answer_groundness(c, a) -> [float, Explanation]
  - For question->answer: f_answer_relevance(q, a) -> [float, Explanation]
We can generalize this to an EdgeEval class
"""
