prompt_template = '''
Use the following piece of information to answer the user's question.
If you don't know the answer, just say you don't know. Don't try to make an answer on yourself.
Context: {context}
question: {question}

Only return the helpful answer below and noting else.
Helpful answer:
'''