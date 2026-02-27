<div align="center">
  <img src="logo.png" alt="DeutschTelcB1Bot" width="120">
  <h1>DeutschTelcB1Bot</h1>
  <p>Adaptive TELC B1 German exam preparation Telegram bot powered by Google Gemini AI.</p>
</div>

## Features

- **Adaptive questions** across all TELC B1 exam areas — difficulty adjusts to your level
- **Spaced repetition (SRS)** — topics due for review surface more often; intervals double on correct answers
- **Question deduplication** — the last 25 questions are remembered so repeats are avoided automatically
- **Daily streaks** — practice every day, earn milestone badges at 3/7/14/30/50/100 days
- **Hints** — tap 🔍 Get Hint for an instant nudge without seeing the answer (up to 2 per question, pre-generated)
- **Voice answers** 🎤 — send a voice message instead of typing; Gemini transcribes and evaluates it in one call
- **Explain More** — after any answer, get a deep AI explanation of the grammar rule
- **Exam simulation** — `/exam` runs a balanced 20-question TELC B1 practice test with a graded summary
- **Progress stats** — `/stats` shows accuracy, avg score, streak, and your top weak areas
- **Inactivity reminders** — hourly nudge (10:00–22:00 UTC) if you haven't practiced in 4 hours
- **Weekly summary** — sent every Sunday with your stats and focus areas

### Topics covered

| Area | Subtopics |
|---|---|
| Grammar | Cases, verb conjugation, modal verbs, prepositions, relative clauses, Konjunktiv II, passive, adjective endings, conjunctions |
| Vocabulary | Daily life, work, travel, health, family, education, shopping, housing, food, culture, environment |
| Reading | Main idea, finding information, text types, inference |
| Writing | Formal letters, informal messages, opinion texts, describing events |

### Bot commands

| Command | Action |
|---|---|
| `/start` | Welcome message + first question |
| `/next` | Get a new practice question immediately |
| `/exam` | Start a 20-question TELC B1 practice test |
| `/stats` | Performance summary, streak, and weak areas |
| `/topic [name]` | Browse topics or focus on a specific one |
| `/help` | Command reference |

---

## Architecture

```
Telegram  ──POST──►  API Gateway  ──►  AWS Lambda (Python 3.12)
                                            │
                               ┌────────────┼────────────┐
                          DynamoDB      Gemini API    EventBridge
                        (3 tables)    (questions +   (reminders)
                                       evaluation)
```

| Component | Service |
|---|---|
| Bot runtime | AWS Lambda |
| HTTP trigger | API Gateway (webhook) |
| User state + streaks | DynamoDB — Users table |
| Session / pending question / exam state | DynamoDB — Sessions table |
| Performance tracking (SRS) | DynamoDB — Performance table |
| Question generation + evaluation | Google Gemini (2.5 Flash) |
| Scheduled reminders | EventBridge hourly rules |

---

## Local development (polling mode)

```bash
# 1. Clone
git clone https://github.com/esrefozturk/DeutschTelcB1Bot.git
cd DeutschTelcB1Bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# fill in TELEGRAM_TOKEN and GEMINI_API_KEY

# 4. Run (uses SQLite locally — no AWS needed)
python bot.py
```

---

## Deployment (AWS Lambda + SAM)

### Prerequisites

- [AWS CLI](https://aws.amazon.com/cli/) configured (`aws configure`)
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey)

### Deploy

```bash
cp .env.example .env
# fill in TELEGRAM_TOKEN and GEMINI_API_KEY

./deploy.sh
```

The script will:
1. Run `sam build` — packages Lambda with all Python dependencies
2. Create an S3 bucket for SAM artifacts
3. Run `sam deploy` — creates the CloudFormation stack with all resources
4. Call `set_webhook.py` — registers the API Gateway URL with Telegram

### Subsequent deploys (after code changes)

Merging to `main` automatically deploys via GitHub Actions — no manual step needed.

For manual deploys or self-hosted forks:

```bash
./deploy.sh --no-guided
```

### View live logs

```bash
aws logs tail /aws/lambda/DeutschTelcBot-prod --follow
```

### Tear down

```bash
./deploy.sh --delete
```

---

## Project structure

```
.
├── bot.py              # Telegram handlers + local polling entry point
├── lambda_handler.py   # AWS Lambda entry point (webhook + EventBridge)
├── gemini_client.py    # Gemini API — question generation, evaluation, voice, hints
├── adaptive.py         # Adaptive learning engine — topic selection, SRS, difficulty
├── database.py         # SQLite backend (local dev)
├── database_dynamo.py  # DynamoDB backend (production)
├── template.yaml       # AWS SAM infrastructure definition
├── deploy.sh           # One-command manual build + deploy + webhook registration
├── set_webhook.py      # Telegram webhook utility
├── requirements.txt    # Python dependencies
├── ruff.toml           # Linter configuration
├── .github/
│   ├── workflows/
│   │   ├── ci.yml      # PR checks: ruff lint + import sanity check
│   │   └── deploy.yml  # Auto-deploy to Lambda on push to main
│   ├── dependabot.yml  # Automated dependency updates
│   └── ISSUE_TEMPLATE/ # Bug report + feature request forms
└── .env.example        # Environment variable template
```

## Environment variables

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | Bot token from [@BotFather](https://t.me/BotFather) |
| `GEMINI_API_KEY` | Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey) |
| `DATABASE_PATH` | SQLite file path (local only, default: `deutsch_bot.db`) |
| `USERS_TABLE` | DynamoDB users table name (Lambda only, set by SAM) |
| `SESSIONS_TABLE` | DynamoDB sessions table name (Lambda only, set by SAM) |
| `PERFORMANCE_TABLE` | DynamoDB performance table name (Lambda only, set by SAM) |
