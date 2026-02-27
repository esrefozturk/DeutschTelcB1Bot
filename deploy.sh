#!/usr/bin/env bash
# deploy.sh — build, deploy, and register the Telegram webhook
#
# Prerequisites:
#   brew install awscli aws-sam-cli   (macOS)
#   aws configure                     (set Access Key / Secret / region)
#
# Usage:
#   ./deploy.sh                      # first deploy (interactive guided mode)
#   ./deploy.sh --no-guided          # subsequent deploys (reuse samconfig.toml)
#   ./deploy.sh --delete             # tear down the whole stack
#   STAGE=staging ./deploy.sh        # deploy to a staging stage
#
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
STAGE="${STAGE:-prod}"
STACK_NAME="${STACK_NAME:-deutsch-telc-bot-${STAGE}}"
AWS_REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || echo eu-central-1)}"
GUIDED="${1:-}"   # pass --no-guided on subsequent deploys

# ── Helpers ───────────────────────────────────────────────────────────────────
red()   { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
blue()  { echo -e "\033[34m$*\033[0m"; }

require() {
  command -v "$1" &>/dev/null || { red "ERROR: '$1' not found. Install it first."; exit 1; }
}

# ── Tear-down shortcut ─────────────────────────────────────────────────────
if [[ "${GUIDED:-}" == "--delete" ]]; then
  blue "==> Removing webhook from Telegram..."
  python set_webhook.py --delete || true

  blue "==> Deleting CloudFormation stack: $STACK_NAME"
  aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$AWS_REGION"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
  green "Stack deleted."
  exit 0
fi

# ── Preflight checks ──────────────────────────────────────────────────────────
require aws
require sam
require python3

# Load .env to get TELEGRAM_TOKEN / GEMINI_API_KEY for parameter-overrides
if [[ -f .env ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

: "${TELEGRAM_TOKEN:?Set TELEGRAM_TOKEN in .env or environment}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in .env or environment}"

# ── Build ─────────────────────────────────────────────────────────────────────
blue "==> Building Lambda package (sam build)..."
sam build --use-container 2>/dev/null || sam build
# --use-container builds inside a Docker image that matches the Lambda runtime.
# Falls back to local build if Docker is not available.

# ── S3 bucket for SAM artifacts ──────────────────────────────────────────────
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
S3_BUCKET="${S3_BUCKET:-sam-artifacts-${ACCOUNT_ID}-${AWS_REGION}}"

if ! aws s3 ls "s3://${S3_BUCKET}" &>/dev/null; then
  blue "==> Creating S3 bucket for SAM artifacts: $S3_BUCKET"
  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws s3 mb "s3://${S3_BUCKET}" --region "$AWS_REGION"
  else
    aws s3 mb "s3://${S3_BUCKET}" --region "$AWS_REGION" \
      --create-bucket-configuration LocationConstraint="$AWS_REGION"
  fi
  aws s3api put-bucket-versioning \
    --bucket "$S3_BUCKET" \
    --versioning-configuration Status=Enabled
fi

# ── Deploy ────────────────────────────────────────────────────────────────────
blue "==> Deploying stack '$STACK_NAME' to region '$AWS_REGION'..."

GUIDED_FLAG=""
[[ "${GUIDED:-}" != "--no-guided" ]] && \
  [[ ! -f samconfig.toml ]] && \
  GUIDED_FLAG="--guided"

sam deploy \
  ${GUIDED_FLAG} \
  --stack-name "$STACK_NAME" \
  --s3-bucket   "$S3_BUCKET" \
  --region      "$AWS_REGION" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    TelegramToken="$TELEGRAM_TOKEN" \
    GeminiApiKey="$GEMINI_API_KEY" \
    Stage="$STAGE" \
  --no-fail-on-empty-changeset \
  --no-confirm-changeset

# ── Fetch outputs ─────────────────────────────────────────────────────────────
blue "==> Fetching stack outputs..."
WEBHOOK_URL=$(
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region     "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='WebhookUrl'].OutputValue" \
    --output text
)

LOG_GROUP=$(
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region     "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='BotFunctionLogGroup'].OutputValue" \
    --output text
)

# ── Register webhook ──────────────────────────────────────────────────────────
blue "==> Registering webhook with Telegram..."
python3 set_webhook.py "$WEBHOOK_URL"

# ── Done ─────────────────────────────────────────────────────────────────────
green ""
green "╔══════════════════════════════════════════════════════════════╗"
green "║  Deployment complete!                                        ║"
green "╠══════════════════════════════════════════════════════════════╣"
green "║  Webhook URL : $WEBHOOK_URL"
green "║  Logs        : aws logs tail $LOG_GROUP --follow"
green "╚══════════════════════════════════════════════════════════════╝"
green ""
