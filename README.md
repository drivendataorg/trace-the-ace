[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://kaiip-platform-prod-public.s3.amazonaws.com/comp_images/NTO_tile_chat.png' width='600'>](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/)

# Trace the Ace

## Goal of the Competition

**This competition focused on evaluating tutoring effectiveness using only the student–tutor conversation.** Participants built models that used tutoring session transcripts to predict whether a student goes on to answer a follow-up question correctly - a practical proxy for whether learning actually took place. Developing better ways of identifying effective tutoring could inform how we train educators, guide real-time support, and expand access to high-quality tutoring through AI-powered tools.

## What's in this Repository

This repository contains code from winning competitors in the [Trace the Ace](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/) challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1 | appleswim (LCBOB and KayX) | 0.5945 | 0.5924 | Combined a fine-tuned Qwen3-8B that parses the full transcript with an XGBoost model trained on 177 hand-crafted features describing tutor moves and student signals
2 | MPWARE | 0.5991 | 0.5947 | Combined a TF-IDF/LightGBM baseline, a 172-feature LightGBM model, and three fine-tuned Qwen readers using logistic regression to learn how much weight to give each model's predictions
3 | Team Chicken (chicken and nicholaschuawork) | 0.5964 | 0.5928 | Blended a fine-tuned Qwen3.5-9B that reads the full transcript with logistic regression on 34 features, including correctness judgments for each student response from a fine-tuned Qwen3.5-2B judge

All winners were chosen based on a combination of leaderboard performance and solution write-up.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: [Productive math talk: a simple reference solution for Trace the Ace](https://drivendata.co/blog/productive-math-talk-reference)**
