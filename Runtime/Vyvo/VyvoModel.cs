using LlamaCpp;
using System;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

namespace VyvoTTS
{
    public class VyvoModel : LlamaCpp.Completion
    {
        protected const int tokeniser_length = 64400;
        protected const int start_of_text = 1;
        protected const int end_of_text = 7;

        protected const int start_of_speech = tokeniser_length + 1;
        protected const int end_of_speech = tokeniser_length + 2;
        protected const int start_of_human = tokeniser_length + 3;
        protected const int end_of_human = tokeniser_length + 4;
        protected const int pad_token = tokeniser_length + 7;

        public override int[] Tokenize(string prompt)
        {
            int[] prompt_token = base.Tokenize(prompt);

            List<int> token_list = new List<int>();

            token_list.Add(start_of_human);
            token_list.AddRange(prompt_token);
            token_list.Add(end_of_text);
            token_list.Add(end_of_human);

            return token_list.ToArray();
        }

        protected override bool EndGeneration(int token, int generated_token_count)
        {
            return Native.llama_vocab_is_eog(_llamaVocab, token) || token == end_of_speech;
        }
    }
}
