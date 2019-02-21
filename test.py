import torch


def test(samples, input_encoder, model_path, device, target_encoder):
    print("--------------------------------------------------------------")

    for sample in samples:
        print("Phrase to translate: " + sample)
        input_tensor = input_encoder.get_encoding_for_sentence_single_tensor(sample)
        beam_search(model_path, input_tensor, device, target_encoder)
        print("--------------------------------------------------------------")


# this methods implements beam search
def beam_search(path, input_tensor, device, target_encoder, beam_max=5):
    encoder_path = path + "encoder.pt"
    decoder_path = path + "decoder.pt"

    encoder = torch.load(encoder_path)
    decoder = torch.load(decoder_path)
    decoder.eval()

    encoder_hidden = encoder.initHidden(device)

    input_length = len(input_tensor)
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    encoder_outputs = torch.zeros(input_length, decoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), 1).unsqueeze(0)
    # decoder_hidden = encoder_hidden[0] + encoder_hidden[1]
    beam = [(0, [], 0, decoder_hidden)]

    for di in range(1, target_encoder.max_length):
        if di < input_length:
            attn_ind = di
        else:
            attn_ind = input_length - 1
        new_beam = []
        for tuple in beam:
            ind = tuple[0]
            if ind != 4:
                log_score = tuple[2]
                previous_tokens = tuple[1]
                tmp = torch.zeros(1, 1, target_encoder.max, device=device)
                tmp[0, 0, ind] = 1.0
                decoder_input = tmp
                decoder_output, decoder_hidden = decoder(decoder_input, tuple[3], encoder_outputs[attn_ind])
                topv, topi = decoder_output.topk(beam_max)
                for i in range(0, beam_max):
                    new_ind = topi[0][i].item()
                    new_log_score = topv[0][i].item()
                    new_tokens = [token for token in previous_tokens]
                    new_tokens.append(new_ind)
                    new_beam.append((new_ind, new_tokens, log_score + new_log_score, decoder_hidden))
            else:
                if len(tuple[1]) > input_length / 2:
                    new_beam.append(tuple)

        beam = prune(new_beam, beam_max)

    for s in target_encoder.get_sentences(beam):
        print(s)
    return


def prune(beam, beam_max):
    if len(beam) <= beam_max:
        return beam
    else:
        sorted_by_score = sorted(beam, key=lambda tup: -tup[2] / len(tup[1]))
        return sorted_by_score[:beam_max]
