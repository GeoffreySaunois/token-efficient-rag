# 🧩 Challenge: Context-Aware Question Answering under Token Constraints

## Objective

Your mission is to design a question-answering (QA) system that intelligently selects and presents the most relevant context to a local language model (LLM) under a strict token budget.
The LLM can only “see” a limited portion of the knowledge base, so your system must make optimal decisions about which pieces of information to include.

## 🎯 Interview Objective

During the discussion, we will focus on:

- Your approach and reasoning,
- The design choices you made,
- Your suggestions for improvement and potential real-world applications.

We are not evaluating the absolute performance of your model, but rather your methodology, clarity of explanation, and ability to present your work.

## 📦 Provided Materials

You will receive:

- A fictional knowledge base (/docs) containing 10 short .md files,
- A list of natural-language questions (questions.json),
- A context token limit of 1024 tokens per question.

## 🧠 Your Tasks

1. Chunking & Retrieval
   - Split the knowledge base into meaningful chunks.
   - Design a retrieval and scoring strategy that ensures only the most relevant text is included within the token budget.
2. Answer Generation
   - Use a small local LLM (1.7B–2B parameters) to generate answers.
   - Ensure your pipeline effectively integrates with the retrieval component.
3. Evaluation & Justification
   - For each question, decide what context to show and why.
   - Explain how and why your approach works (or where it might fail).
   - Optionally, suggest metrics or manual checks to evaluate performance.

📁 Deliverables
Your submission should include:

1. GitHub Repository
   - Clean, well-documented code
   - Clear execution instructions
   - Must be shared at least one day before the interview
2. Presentation (PowerPoint or PDF)
   - Your approach and design rationale
   - Results, model choices, and key challenges
   - Areas for improvement and future development ideas
   - (Bonus) A deployment architecture diagram if you were to productionize the solution
