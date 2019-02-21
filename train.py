import torch
import time
import math
import matplotlib.ticker as ticker
import random
from decoder_rnn import DecoderRNN
from attn_decoder_rnn import AttnDecoderRNN
from simple_attn_decoder import SimpleAttnDecoderRNN
from encoder_rnn import EncoderRNN
from pathlib import Path
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, max_length):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = len(input_tensor)
    target_length = len(target_tensor)

    encoder_outputs = torch.zeros(max_length, decoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    # encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    loss = 0

    decoder_input = target_tensor[0]
    # decoder_hidden = encoder_hidden[0] + encoder_hidden[1]
    decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]), 1)
    decoder_hidden = decoder_hidden.unsqueeze(0)

    use_teacher_forcing = True if random.random() < 1.95 else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(1, target_length):
            # decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            if di < input_length:
                ind = di
            else:
                ind = input_length - 1
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs[ind])
            decoder_input = target_tensor[di]

            c = torch.max(target_tensor[di][0], 1)[1]
            loss += criterion(decoder_output, c)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)

            ind = topi.detach().item()
            tmp = torch.zeros(1, 1, decoder_input.shape[2], device=device)
            tmp[0, 0, ind] = 1.0

            c = torch.max(target_tensor[di][0], 1)[1]
            loss += criterion(decoder_output, c)
            if ind == 4:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_iters(path, n_iters, input_texts, target_texts, input_encoder, target_encoder, device, max_length, print_every=100, plot_every=200, learning_rate=0.0001, momentum=0.99):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    latent_dim = 512  # Latent dimensionality of the encoding space.

    encoder_path = path + "encoder.pt"
    decoder_path = path + "decoder.pt"

    encoder = torch.load(encoder_path) if Path(encoder_path).exists() else EncoderRNN(input_encoder.max, latent_dim).to(device)
    decoder = torch.load(decoder_path) if Path(decoder_path).exists() else SimpleAttnDecoderRNN(latent_dim, target_encoder.max).to(device)
    # decoder = torch.load(decoder_path) if Path(decoder_path).exists() else AttnDecoderRNN(latent_dim, target_encoder.max, max_length).to(device)

    # encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), betas=(momentum, 0.999), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), betas=(momentum, 0.999), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    for epoch in range(0, 5):
        print("epoch: " + str(epoch) + '\n')
        for iter in range(0, n_iters + 1):
            input_tensor = input_encoder.get_encoding_for_sentence_single_tensor(input_texts[iter])
            target_tensor = target_encoder.get_encoding_for_sentence(target_texts[iter])

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, max_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        torch.save(encoder, encoder_path)
        torch.save(decoder, decoder_path)

    torch.save(encoder_optimizer.state_dict(), path + "encoder_optimizer.pt")
    torch.save(decoder_optimizer.state_dict(), path + "decoder_optimizer.pt")

    showPlot(plot_losses)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent + 1)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)