from langchain import hub 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

class GraderUtils:
    def __init__(self, model):
        self.model = model

    def create_retrieval_grader(self):
        "Creates a retrieval grader that assess the relevance of a retrieved document to a user question"
        grade_prompt = PromptTemplate(
            template="""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binnary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>

            Here is the retrieveAd document: \n\n {document} \n\n
            Here is the user question: {input} \n
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=['document', 'input']
        )
        retriever_grader = grade_prompt | self.model | JsonOutputParser()
        return retriever_grader
    
    def create_hallucination_grader(self):
        "Creates a hallucination grader that assess whether an answer is grounded in/supported by a set of facts."
        hallucination_prompt = PromptTemplate(
            template="""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ----- \n
            {documents}
            \n ----- \n 
            Here is the answer: {generation}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables = ['documents', 'generation']
        )
        hallucination_grader = hallucination_prompt | self.model | JsonOutputParser()
        return hallucination_grader 
    
    def create_code_evaluator(self):
        "Creates a code evaluator that assess whether the generated code is correct and relevant to the given question."
        eval_template = PromptTemplate(
            template="""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a code evaluator assessing whether the generated code is correct and relevant to the given question.
            Provide a JSON response with the following keys:

            'score': A binary score 'yes' or 'no' indicating whether the code is correct and relevant.
            'feedback': A brief explanation of your evaluation, including any issues or improvements needed.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the generated code:
            \n ----- \n 
            {generation}
            \n ----- \n
            Here is the question: {input}
            \n ----- \n 
            Here are the relevant documents: {documents}
            <|begin_of_text|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=['generation', 'input', 'documents']
        )
        code_evaluator = eval_template | self.model | JsonOutputParser()
        return code_evaluator
    
    def create_question_rewriter(self):
        "Creates a question rewriter chain that rewrites a given question to improve its clarity and relevance."
        re_write_prompt = hub.pull("efriis/self-rag-question-rewriter")
        question_rewriter = re_write_prompt | self.model | StrOutputParser()
        return question_rewriter
    