
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# 1. Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=500
)

parser = StrOutputParser()


# 2. Branch 1: Generate Notes
notes_prompt = ChatPromptTemplate.from_template("""
You are a teacher.

Create concise study notes on this topic:
{topic}

Include:
1. Definition
2. Key concepts
3. Important points
4. One practical example

Keep the notes concise.
""")

notes_chain = notes_prompt | llm | parser


# 3. Branch 2: Generate Quiz
quiz_prompt = ChatPromptTemplate.from_template("""
You are a quiz generator.

Create 5 multiple-choice questions about:
{topic}

For each question provide:
- Question
- Four options (A, B, C, D)
- Correct answer

Keep the questions concise.
""")

quiz_chain = quiz_prompt | llm | parser


# 4. Parallel execution: Both branches run together
parallel_chain = RunnableParallel(
    notes=notes_chain,
    quiz=quiz_chain
)


# 5. Final Module: Combine both outputs
final_prompt = ChatPromptTemplate.from_template("""
You are an educational content organizer.

Combine the following notes and quiz into one
well-organized learning resource.

TOPIC: {topic}

STUDY NOTES:
{notes}

QUIZ:
{quiz}

Organize the final output using these sections:
1. Topic Overview
2. Study Notes
3. Practice Quiz
4. Answer Key

Preserve the important information.
Avoid unnecessary repetition.
""")

final_chain = final_prompt | llm | parser


# 6. ONE complete chain
chain = (
    RunnablePassthrough.assign(
        parallel_results=parallel_chain
    )
    | RunnablePassthrough.assign(
        final_output=lambda x: final_chain.invoke({
            "topic": x["topic"],
            "notes": x["parallel_results"]["notes"],
            "quiz": x["parallel_results"]["quiz"]
        })
    )
)


# 7. Get user input
topic = input("Enter a topic: ")

chain.get_graph().print_ascii()
# 8. Run the complete architecture
result = chain.invoke({"topic": topic})


# 9. Display outputs
print("\n========== NOTES ==========")
print(result["parallel_results"]["notes"])

print("\n========== QUIZ ==========")
print(result["parallel_results"]["quiz"])

print("\n========== FINAL OUTPUT ==========")
print(result["final_output"])


