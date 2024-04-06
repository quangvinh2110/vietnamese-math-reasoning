set -ex

python download_model.py \
    --repo_id "vilm/vinallama-7b-chat" \
    --local_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/zac2023/pretrained_models/vinallama-7b-chat" \
    --cache_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/cache/vinallama-7b-chat"

python download_model.py \
    --repo_id "Viet-Mistral/Vistral-7B-Chat" \
    --local_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/zac2023/pretrained_models/Vistral-7B-Chat" \
    --cache_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/cache/Vistral-7B-Chat"

python download_model.py \
    --repo_id "meta-math/MetaMath-Mistral-7B" \
    --local_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/zac2023/pretrained_models/MetaMath-Mistral-7B" \
    --cache_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/cache/MetaMath-Mistral-7B"

python download_model.py \
    --repo_id "WizardLM/WizardMath-7B-V1.1" \
    --local_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/zac2023/pretrained_models/WizardMath-7B-V1.1" \
    --cache_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/cache/WizardMath-7B-V1.1"

python download_model.py \
    --repo_id "Qwen/Qwen1.5-7B-Chat" \
    --local_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/zac2023/pretrained_models/Qwen1.5-7B-Chat" \
    --cache_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/cache/Qwen1.5-7B-Chat"

python download_model.py \
    --repo_id "llm-agents/tora-code-7b-v1.0" \
    --local_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/zac2023/pretrained_models/tora-code-7b-v1.0" \
    --cache_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/cache/tora-code-7b-v1.0"

python download_model.py \
    --repo_id "WizardLM/WizardCoder-Python-7B-V1.0" \
    --local_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/zac2023/pretrained_models/WizardCoder-Python-7B-V1.0" \
    --cache_dir "/data2/cmdir/home/ioit101/Dang_DHCN/projects/LLM_2023/Zalo_Challenge/vinhnq29/cache/WizardCoder-Python-7B-V1.0"



