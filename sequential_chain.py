from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b",
               max_tokens=700)

parser = StrOutputParser()


profile_prompt = ChatPromptTemplate.from_template("""
You are an AI career advisor.

Analyze the following user profile.

Current Skills: {skills}
Career Goal: {goal}
Available Study Hours Per Day: {hours}

Provide:
1. Current skill assessment
2. Important skill gaps
3. Learning priorities
4. Estimated difficulty level
""")

profile_chain = profile_prompt | llm | parser



roadmap_prompt = ChatPromptTemplate.from_template("""
You are an expert technical mentor.

Create a structured learning roadmap using this profile analysis:

{profile_analysis}

Career Goal: {goal}
Daily Study Hours: {hours}

Include:
1. Learning phases in correct order
2. Topics to learn in each phase
3. Practical exercises
4. Estimated duration of each phase
5. Prerequisites between topics

Avoid unnecessary topics.
""")

roadmap_chain = roadmap_prompt | llm | parser



project_prompt = ChatPromptTemplate.from_template("""
You are a senior AI engineer and project mentor.

Based on this learning roadmap:

{learning_roadmap}

And the user's career goal:

{goal}

Suggest 3 practical projects.

For each project, provide:
1. Project title
2. Problem statement
3. Required technologies
4. Main features
5. Implementation steps
6. How the project helps in interviews

Projects should progress from intermediate to advanced.
""")

project_chain = project_prompt | llm | parser



study_plan_prompt = ChatPromptTemplate.from_template("""
You are an AI learning planner.

Create a practical weekly study plan using the following information.

PROFILE ANALYSIS:
{profile_analysis}

LEARNING ROADMAP:
{learning_roadmap}

PROJECT IDEAS:
{project_ideas}

Daily Available Hours: {hours}
Career Goal: {goal}

Generate:
1. A 7-day study timetable
2. Daily theory and coding tasks
3. Weekly revision strategy
4. Project milestones
5. A self-assessment checklist
6. Clear completion criteria

Make the plan realistic and actionable.
""")

study_plan_chain = study_plan_prompt | llm | parser



def generate_course_plan(skills, goal, hours):

    # Chain 1
    profile_analysis = profile_chain.invoke({
        "skills": skills,
        "goal": goal,
        "hours": hours
    })

    # Chain 2
    learning_roadmap = roadmap_chain.invoke({
        "profile_analysis": profile_analysis,
        "goal": goal,
        "hours": hours
    })

    # Chain 3
    project_ideas = project_chain.invoke({
        "learning_roadmap": learning_roadmap,
        "goal": goal
    })

    # Chain 4
    final_study_plan = study_plan_chain.invoke({
        "profile_analysis": profile_analysis,
        "learning_roadmap": learning_roadmap,
        "project_ideas": project_ideas,
        "hours": hours,
        "goal": goal
    })

    return {
        "profile_analysis": profile_analysis,
        "learning_roadmap": learning_roadmap,
        "project_ideas": project_ideas,
        "final_study_plan": final_study_plan
    }



if __name__ == "__main__":

    skills = input("Enter your current skills: ")
    goal = input("Enter your career goal: ")
    hours = input("Daily study hours: ")

    result = generate_course_plan(skills, goal, hours)

    print("\n" + "=" * 60)
    print("PROFILE ANALYSIS")
    print("=" * 60)
    print(result["profile_analysis"])

    print("\n" + "=" * 60)
    print("LEARNING ROADMAP")
    print("=" * 60)
    print(result["learning_roadmap"])

    print("\n" + "=" * 60)
    print("PROJECT IDEAS")
    print("=" * 60)
    print(result["project_ideas"])

    print("\n" + "=" * 60)
    print("FINAL WEEKLY STUDY PLAN")
    print("=" * 60)
    print(result["final_study_plan"])
