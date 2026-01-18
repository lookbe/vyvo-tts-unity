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

        protected class VyvoPayload : CompletionPayload
        {
            public string Prompt;
        }

        public override void Prompt(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
            {
                return;
            }

            if (_llamaContext == IntPtr.Zero)
            {
                Debug.LogError("invalid context");
                return;
            }

            if (_llamaModel == IntPtr.Zero)
            {
                Debug.LogError("model not loaded");
                return;
            }

            if (status != LlamaCpp.ModelStatus.Ready)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = LlamaCpp.ModelStatus.Generate;
            var payload = new VyvoPayload()
            {
                Prompt = prompt,

                Temp = this.temperature,
                TopK = this.topK,
                TopP = this.topP,
                MinP = this.minP,
                RepeatPenalty = this.repeatPenalty,

            };

            RunBackground(payload, RunMayaPrompt);
        }

        void RunMayaPrompt(VyvoPayload inputPayload, CancellationToken cts)
        {
            int BOS_TOKEN = LlamaCpp.Native.llama_vocab_bos(_llamaVocab);
            int[] prompt_token = Tokenize(inputPayload.Prompt);

            List<int> token_list = new List<int>();
            token_list.Add(start_of_human);
            token_list.AddRange(prompt_token);
            token_list.Add(end_of_text);
            token_list.Add(end_of_human);

            var payload = new GenerationPayload()
            {
                Tokens = token_list.ToArray(),
                Temp = inputPayload.Temp,
                TopK = inputPayload.TopK,
                TopP = inputPayload.TopP,
                MinP = inputPayload.MinP,
                RepeatPenalty = inputPayload.RepeatPenalty,

            };

            RunGenerate(payload, cts);
        }

        protected override bool EndGeneration(int token, int generated_token_count)
        {
            return Native.llama_vocab_is_eog(_llamaVocab, token) || token == end_of_speech;
        }
    }
}
