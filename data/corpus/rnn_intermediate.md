## Page 1

# Recurrent Neural Networks (RNN)

## Sequential Modeling and Hidden State

Recurrent neural networks are designed to process sequential data such as time series, speech, and natural language. Unlike feedforward networks, RNNs maintain a hidden state that captures information from previous time steps. At each step, the network combines the current input with the previous hidden state to produce an updated representation. This recursive structure allows RNNs to model temporal dependencies and contextual relationships in sequences. Applications include language modeling, machine translation, and stock price forecasting. However, training RNNs can be challenging due to gradient instability across long sequences.

## Vanishing and Exploding Gradient Problems

When training recurrent networks over long sequences, gradients propagated backward through time can become extremely small or excessively large. The vanishing gradient problem prevents earlier time steps from influencing learning, limiting the model's ability to capture long-range dependencies. Conversely, exploding gradients can cause unstable updates and numerical overflow. Techniques such as gradient clipping, careful weight initialization, and specialized architectures help mitigate these issues. Understanding gradient behavior is essential for effectively training deep sequence models.

## LSTM and GRU Architectures

Long Short-Term Memory networks and Gated Recurrent Units are advanced recurrent architectures designed to address gradient issues. LSTMs introduce memory cells and three gating mechanisms — input, forget, and output gates — that regulate information

---


## Page 2

flow across time steps. GRUs simplify this structure by combining gates while retaining strong performance. These gated mechanisms allow networks to preserve relevant long-term context while discarding noise. As a result, LSTMs and GRUs are widely used in speech recognition, text generation, and recommendation systems where temporal understanding is critical.

