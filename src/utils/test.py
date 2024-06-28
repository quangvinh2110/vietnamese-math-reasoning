  for sample in train_data:
        system_prompt = random.choice(SYS_PREFIX)+"\n"+SYS_SUFFIX
        
        instruction = random.choice(vi_instruction_multiple_choices_prompts)
        question = sample["md_question"]
        choices = sample["md_choices"]
        choices = [x for x in choices if x]
        choices.sort(key=lambda choice: choice[0])
        choices="\n".join(choices)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            instruction=instruction,
            question=question,
            choices=choices
        ).strip()
        
        answer = sample["instruct_code"].strip()
        if len(answer) > 4000:
            continue
        if not (question and choices and answer):
            continue
        answer_parts = OUTPUT_PATTERN.split(answer)
        if len(answer_parts) == 3:
            input_output_data.append({"segments": [
                {
                    "label": False,
                    "text": tokenizer.apply_chat_template([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ], tokenize=False, add_generation_prompt=True)
                },
                {
                    "label": True,
                    "text": answer_parts[0]+"```output"
                },
                {
                    "label": False,
                    "text": answer_parts[1]+"```"
                },
                {
                    "label": True,
                    "text": answer_parts[2] + tokenizer.eos_token
                }
            ]})
        elif len(answer_parts) == 1:
            input_output_data.append({"segments": [
                {
                    "label": False,
                    "text": tokenizer.apply_chat_template([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ], tokenize=False, add_generation_prompt=True)
                },
                {
                    "label": True,
                    "text": answer + tokenizer.eos_token
                }
            ]})
        else:
            continue
        

    with open(args.output_file, "w") as f:
        f.write(json.dumps(
            input_output_data,
            indent=4,
            ensure_ascii=False
        ))

    # input_output_dataset = load_dataset(
    #     "json", 
    #     data_files=args.output_file,
    #     split="train"
    # )

    # input_output_dataset.push_to_hub(
    #     "vinhnq29/ViMathQA",
    #     "train_v1", split=f"input_output_{normalize_name(args.model_name)}",
    #     token="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"
    # )
