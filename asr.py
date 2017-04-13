import tensorflow as tf

def sliding_window(inp, frame_length, frame_shift, padding="VALID", name=None):
    '''
    Runs a sliding window across a 1D signal (of audio data, for example).
    Params:
      - inp:  A [batch_size, signal_length] tensor
      - frame_length:  The length of each frame in number of samples
      - frame_shift:  How many samples to shift the window
      - padding:  How to pad the ends, can be "SAME" or "VALID"
    Returns:
      - A [batch_size, signal_length // frame_shift (approx, depending on padding), frame_length] tensor of frames

    Example:

    ```
    p = tf.placeholder(tf.float32, [None, None])
    patches = sliding_window(p, 5, 3, padding="VALID")

    with tf.Session() as sess:
        x = sess.run(patches, feed_dict={
            p: np.vstack([np.arange(15), np.arange(16, 16 + 15)])
        })
        print(x)
    ```

    Prints:
    [[[  0.   1.   2.   3.   4.]
      [  3.   4.   5.   6.   7.]
      [  6.   7.   8.   9.  10.]
      [  9.  10.  11.  12.  13.]]

     [[ 16.  17.  18.  19.  20.]
      [ 19.  20.  21.  22.  23.]
      [ 22.  23.  24.  25.  26.]
      [ 25.  26.  27.  28.  29.]]]
    '''
    assert(len(inp.shape) == 2)  # [batch_size, frame length]
    with tf.name_scope(name or "sliding_window"):
        expanded = tf.expand_dims(tf.expand_dims(inp, 2), 3)
        frames = tf.extract_image_patches(expanded, [1, frame_length, 1, 1], [1, frame_shift, 1, 1], [1, 1, 1, 1], padding)
        frames = tf.squeeze(frames, [2]) 
    return frames
