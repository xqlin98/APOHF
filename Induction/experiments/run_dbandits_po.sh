# datasets=(auto_debugging odd_one_out cause_and_effect common_concept word_sorting active_to_passive antonyms auto_categorization diff first_word_letter informal_to_formal larger_animal letters_list negation num_to_verbal object_counting orthography_starts_with periodic_elements rhymes second_word_letter sentence_similarity sentiment singular_to_plural sum synonyms taxonomy_animal translation_en-de translation_en-es translation_en-fr word_unscrambling)
datasets='auto_debugging'

python experiments/run_dbandits_po.py \
--task ${datasets} \
--n_prompt_tokens 5 \
--nu 1 \
--lamdba 0.1 \
--n_init 5 \
--n_domain 200 \
--total_iter 150 \
--local_training_iter 1000 \
--n_eval 1000 \
--intrinsic_dim 10 \
--gpt gpt-3.5-turbo-1106 \
--func neural \
--name PO

echo ${commands[$1]} 
eval ${commands[$1]}
