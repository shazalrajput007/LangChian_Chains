from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

# 1. Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=100
)

parser = StrOutputParser()


# 2. Feedback classification prompt
classification_prompt = ChatPromptTemplate.from_template("""
Analyze the customer's feedback.

Feedback:
{feedback}

Classify it as exactly one of these labels:
positive
negative

Rules:
- Positive: The customer expresses satisfaction or appreciation.
- Negative: The customer expresses dissatisfaction, a complaint,
  or a problem.

Return only one label: positive or negative.
""")

classification_chain = classification_prompt | llm | parser


# 3. Positive feedback prompt
positive_prompt = ChatPromptTemplate.from_template("""
You are a friendly customer support representative.

Customer feedback:
{feedback}

Write a warm, professional response that:
- Thanks the customer for the positive feedback.
- Shows appreciation.
- Encourages them to continue using the service.

Keep it concise and natural.
""")

positive_chain = positive_prompt | llm | parser


# 4. Negative feedback prompt
negative_prompt = ChatPromptTemplate.from_template("""
You are a professional customer support representative.

Customer feedback:
{feedback}

Write an empathetic response that:
- Apologizes for the customer's bad experience.
- Acknowledges their specific problem.
- Offers help and asks for any details needed to resolve it.
- Avoids claiming that the issue is already fixed.

Keep it polite and concise.
""")

negative_chain = negative_prompt | llm | parser


# 5. Conditional branch
conditional_chain = RunnableBranch(
    (
        lambda x: x["sentiment"].strip().lower() == "positive",
        positive_chain
    ),
    (
        lambda x: x["sentiment"].strip().lower() == "negative",
        negative_chain
    ),
    # Default branch for unexpected classification
    negative_chain
)


# 6. Complete workflow
def generate_support_response(feedback):

    sentiment = classification_chain.invoke({
        "feedback": feedback
    })

    result = conditional_chain.invoke({
        "feedback": feedback,
        "sentiment": sentiment
    })

    return {
        "feedback": feedback,
        "sentiment": sentiment,
        "response": result
    }


# 7. User input
if __name__ == "__main__":

    feedback = input("Enter customer feedback: ")

    result = generate_support_response(feedback)

    print("\n========== CUSTOMER FEEDBACK ==========")
    print(result["feedback"])

    print("\n========== SENTIMENT ==========")
    print(result["sentiment"])

    print("\n========== SUPPORT RESPONSE ==========")
    print(result["response"])
