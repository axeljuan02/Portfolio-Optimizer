from openai import OpenAI

client = OpenAI()





def ask_ai(question=None, messages=None):
    if messages is None:        # cas 1 : on construit nous-mêmes les messages, function API de base avec initialisation (1er dictionnaire) + question simple
        messages = [
            {"role": "system", "content": "You are SORA AI, a professional and pedagogical financial assistant. "
                                          "You can explain models, financial concepts, and guide investors clearly. "
                                          "If data is missing, use general financial knowledge to explain or give advice, "
                                          "but make clear when you are making general statements."},
            {"role": "user", "content": question}
        ]
    # cas 2 : si messages est fourni (messages=chat_history), on lui donne l'initialisation + historique avec question
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
    )
    return response.choices[0].message.content





def chatbox_ai(master_summary):
    """
    Function to get an interactive chatbox for further explanation with SORA AI.
    Different color feature for both the user (blue) and SORA AI (basic white) in the terminal.
    Memory (chat_history) feature for more precise & appropriate answers by SORA AI.
    """
    blue = '\033[94m'
    reset = '\033[0m'

    chat_history = [
        {"role": "system", "content": "You are SORA AI, a professional and pedagogical financial assistant. When answering questions, always reference the data provided (exact numbers, percentages, and metrics) whenever relevant. Do not replace them with generic explanations.Do not use LaTeX, markdown math, or special symbols like \\( ... \\); write everything in plain text for a terminal display."},
        {f"role": "user", "content": f"Here are the portfolio analysis results to use for all answers: {master_summary}"}
    ]

    print(f"{blue}Don't mind to chat with SORA AI for further explanation.{reset}")

    while True:
        chat = input(f"{blue}Your question : {reset}")

        if chat.lower() in ['exit', 'quit', 'bye']:
            break

        chat_history.append({"role": "user", "content": chat})
        ai_response = ask_ai(messages=chat_history)
        chat_history.append({"role": "assistant", "content": ai_response})

        print(f"SORA AI : {ai_response}")





investor_prompt = """
You are a financial assistant. Your goal is to help an amateur-to-intermediate investor understand the results of portfolio analysis models.

Here is the summary of the computed results: {master_summary}

Write your answer in **three clear sections**:

1 **Model Explanation** - Briefly explain the model(s) used (e.g. Efficient Frontier, Monte Carlo Simulation, Fama-French 5) in simple, accessible terms, so the investor understands what each model does and why it matters.

2 **Data Overview** - Present:
   - The investor's **initial data** (from the input before optimization)
   - The **final recommended data** (from the model's results)
   - Be factual and do NOT invent or assume any missing information.

3 **Interpretation & Guidance**  Explain what these results mean for the investor. Summarize the key insights and implications in clear, actionable language, without speculation or generic financial advice.

Rules:
- Do NOT invent numbers, models, or assumptions beyond the provided data.
- Keep the tone friendly but professional, as if explaining to someone with some knowledge of investing but not an expert.
- Use plain English and short paragraphs, avoid jargon or overly technical terms.
- You only EXPLAIN the results provided.
- Do NOT add any generic analysis or external context.
- If data is missing, simply say 'Missing data'.
"""
