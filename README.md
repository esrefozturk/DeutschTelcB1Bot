# DeutschTelcB1Bot

Adaptive TELC B1 German exam preparation Telegram bot powered by Google Gemini AI.

## What it does

- Sends practice questions covering all TELC B1 exam areas
- Evaluates your answers and gives instant feedback
- Adapts to your weak spots — topics you get wrong appear more often
- Automatically increases difficulty as you improve
- Tracks your progress with `/stats`

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
| `/start` | Welcome + first question |
| `/next` | Skip to a new question |
| `/pause` | Pause practice |
| `/resume` | Resume practice |
| `/stats` | Performance summary + weak areas |
| `/topic grammar cases` | Focus on a specific topic/subtopic |
| `/help` | Command reference |

---

## Architecture

```
Telegram  ──POST──►  API Gateway  ──►  AWS Lambda (Python 3.12)
                                            │
                               ┌────────────┼────────────┐
                          DynamoDB      Gemini API    (state)
                        (3 tables)    (questions +
                                       evaluation)
```

| Component | Service |
|---|---|
| Bot runtime | AWS Lambda (`DeutschTelcBot-prod`) |
| HTTP trigger | API Gateway (`/prod/webhook`) |
| User state | DynamoDB — `DeutschBotUsers-prod` |
| Session / pending question | DynamoDB — `DeutschBotSessions-prod` |
| Performance tracking | DynamoDB — `DeutschBotPerformance-prod` |
| Question generation | Google Gemini 2.0 Flash |

---

## Local development (polling mode)

```bash
# 1. Clone
git clone git@github.com:esrefozturk/DeutschTelcB1Bot.git
cd DeutschTelcB1Bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# fill in TELEGRAM_TOKEN and GEMINI_API_KEY

# 4. Run (uses SQLite locally, no AWS needed)
python bot.py
```

---

## Deployment (AWS Lambda + SAM)

### Prerequisites

```bash
brew install awscli aws-sam-cli
aws configure   # enter Access Key ID, Secret, region (us-east-1)
```

### First deploy

```bash
cp .env.example .env
# fill in TELEGRAM_TOKEN and GEMINI_API_KEY

./deploy.sh
```

The script will:
1. Run `sam build` — packages Lambda with all Python dependencies
2. Create an S3 bucket for SAM artifacts (`sam-artifacts-<account>-<region>`)
3. Run `sam deploy` — creates the CloudFormation stack with all resources
4. Call `set_webhook.py` — registers the API Gateway URL with Telegram

### Subsequent deploys (after code changes)

```bash
./deploy.sh --no-guided
```

### Tear down

```bash
./deploy.sh --delete
```

### Manual webhook management

```bash
python set_webhook.py https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/webhook
python set_webhook.py --info
python set_webhook.py --delete
```

### View live logs

```bash
aws logs tail /aws/lambda/DeutschTelcBot-prod --follow
```

---

## Deployed resources

| Resource | Name / ARN |
|---|---|
| CloudFormation stack | `deutsch-telc-bot-prod` |
| Lambda function | `DeutschTelcBot-prod` (us-east-1) |
| API Gateway | `DeutschTelcBotApi-prod` |
| Webhook URL | `https://zs8hd2qa28.execute-api.us-east-1.amazonaws.com/prod/webhook` |
| Log group | `/aws/lambda/DeutschTelcBot-prod` |

---

## Project structure

```
.
├── bot.py              # Telegram handlers + local polling entry point
├── lambda_handler.py   # AWS Lambda entry point (webhook mode)
├── gemini_client.py    # Gemini API — question generation & answer evaluation
├── adaptive.py         # Adaptive learning engine — topic selection & difficulty
├── database.py         # SQLite backend (local dev)
├── database_dynamo.py  # DynamoDB backend (Lambda / production)
├── template.yaml       # AWS SAM infrastructure definition
├── deploy.sh           # One-command build + deploy + webhook registration
├── set_webhook.py      # Telegram webhook utility
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variable template
```

## Environment variables

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | Bot token from [@BotFather](https://t.me/BotFather) |
| `GEMINI_API_KEY` | Google Gemini API key from [AI Studio](https://aistudio.google.com/apikey) |
| `DATABASE_PATH` | SQLite file path (local only, default: `deutsch_bot.db`) |
| `USERS_TABLE` | DynamoDB users table name (Lambda only) |
| `SESSIONS_TABLE` | DynamoDB sessions table name (Lambda only) |
| `PERFORMANCE_TABLE` | DynamoDB performance table name (Lambda only) |
