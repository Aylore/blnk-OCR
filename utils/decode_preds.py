


def decode_prediction(logits, 
                      label_converter):
    tokens = logits.softmax(2).argmax(2)
    tokens = tokens.squeeze(1).numpy()
    
    # convert tor stings tokens
    tokens = ''.join([label_converter.idx2char[token] 
                      if token != 0  else '-' 
                      for token in tokens])
    tokens = tokens.split('-')
    
    # remove duplicates
    text = [char 
            for batch_token in tokens 
            for idx, char in enumerate(batch_token)
            if char != batch_token[idx-1] or len(batch_token) == 1]
    text = ''.join(text)
    return text