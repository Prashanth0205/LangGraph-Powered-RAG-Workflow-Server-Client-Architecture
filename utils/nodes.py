from utils.generate_chain import create_generate_chain

class GraphNodes:
    def __init__(self, llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter):
        self.llm = llm 
        self.retriever = retriever
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader =  hallucination_grader
        self.code_evaluator = code_evaluator
        self.question_rewriter = question_rewriter
        self.generate_chain = create_generate_chain(llm)

    def retrieve(self, state):
        "Retrieve documents"
        print("-----RETRIEVE-----")
        question = state['input']
        documents = self.retriever.invoke(question)
        return {"documents": documents, "input": question}
    
    def generate(self, state):
        "Generate answer"
        print("-----GENERATE-----")
        question = state['input']
        documents = state['documents']
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        return {'documents': documents, "input": question, "generation": generation}
    
    def grade_documents(self, state):
        "Determines whether the retrieved documents are relevant to the question"
        print("-----CHECK DOCUMENT RELEVANCE TO THE QUESTION-----")
        question = state['input']
        documents = state['documents']

        filtered_documents = []
        for d in documents:
            score = self.retrieval_grader.invoke({'input': question, 'document': d.page_content})
            grade = score['score']
            if grade == "yes":
                print("-----GRADE: DOCUMENT RELEVANT-----")
                filtered_documents.append(d)
            else:
                print("-----GRADE: DOCUMENT IR-RELEVANT-----")
                continue
        return {"documents": filtered_documents, "input": question}
    
    def transform_query(self, state):
        "Transform the query to produce a better question"
        print(f"-----TRANSFORM QUERY-----")
        question = state['input']
        documents = state['documents']
        better_question = self.question_rewriter.invoke({'question': question})
        return {"documents": documents, "input": better_question}
