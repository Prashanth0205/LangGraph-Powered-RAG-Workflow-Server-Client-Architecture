import streamlit as st
from langserve import RemoteRunnable 
from pprint import pprint

st.title("Welcome to Pytorch Chat-bot App")
input_text = st.text_input("Ask Pytorch related questions here!")

if input_text:
    with st.spinner("Processing..."):
        try:
            app = RemoteRunnable("http://localhost:8000/torch_chat/")
            for output in app.stream({"input": input_text}):
                for key, value in output.items():
                    # Node
                    pprint(f"Node '{key}':")
                    # Optional: print full state at each node
                    # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
                pprint("\n---\n")
            output = value['generation']
            st.write(output)

        except Exception as e:
            st.error(f"Error: {e}")

# def get_response(input_text):
#     app = RemoteRunnable("http://localhost:8000/speckle_chat/")
#     for output in app.stream({"input": input_text}):
#         for key, value in output.items():
#             # Node
#             pprint(f"Node '{key}':")
#             # Optional: print full state at each node
#             # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
#         pprint("\n---\n")
#     output = value['generation']
#     return output  

# import gradio as gr
# from langserve import RemoteRunnable
# from pprint import pprint

# # Create the UI In Gradio
# iface = gr.Interface(fn=get_response, 
#           inputs=gr.Textbox(
#           value="Enter your question"), 
#           outputs="textbox",  
#           title="Q&A over Speckle's developer docs",
#           description="Ask a question about Speckle's developer docs and get an answer from the code assistant. This assistant looks up relevant documents and answers your code-related question.",
#           examples=[["How do I install Speckle's python sdk?"], 
#                   ["How to commit and retrieve an object from Speckle?"],
#                   ],
#           theme=gr.themes.Soft(),
#           allow_flagging="never",)

# iface.launch(share=True) # put share equal to True for public URL