
class EdgeGraph:
    def __init__(self, hallucination_grader, code_evaluator):
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator

    def decide_to_generate(self, state):
        "Determines whether to generate an answer, or re-generator a question"
        print(f"-----ASSESS GRADED DOCUMENTS-----")
        question = state['input']
        filtered_documents = state['documents']

        if not filtered_documents:
            # All documents have been filtered check relevance 
            # We will re-generate a new query 
            print("-----DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY-----")
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("-----DECISION: GENERATE-----")
            return "generate"
        
    def grade_generation_v_documents_and_question(self, state):
        "Determiens whether the generation is grounded in the document and answers question"
        print("-----CHECK HALLUCINATIONS-----")
        question = state['input']
        documents = state['documents']
        generation = state['generation']

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']

        # Check hallucination 
        if grade == 'yes':
            print("-----DECISION: GENERATION IS GROUNDED IN DOCUMENTS-----")
            # Check question-answering 
            score = self.code_evaluator.invoke({"input": question, "generation": generation, "documents": documents})
            grade = score['score']
            if grade == "yes":
                print("-----DECISION: GENERATION ADDRESSESS QUESTION-----")
                return "useful"
            else:
                print(f"-----DECISION: GENERATION DOES NOT ADDRESS QUESTION-----")
                return "not useful"
        else:
            print(f"-----DECISION: GENERATIONS ARE HALLUCINATED, RE-TRY-----")
            return "not supported"