"""
DOCSTRING
"""
import glob
import midi
import numpy
import tensorflow
import tqdm

class MergeSamples:
    """
    DOCSTRING
    """
    def __call__(self):
        try:
            files = glob.glob('generated*.mid*')
        except Exception as exception:
            raise exception
        songs = numpy.zeros((0, 156))
        for f in tqdm.tqdm(files):
            try:
                song = numpy.array(MidiManipulation.midi_to_note_state_matrix(f))
                if numpy.array(song).shape[0] > 10:
                    songs = numpy.concatenate((songs, song))
            except Exception as exception:
                raise exception
        print('samples merging ...')
        print(numpy.shape(songs))
        MidiManipulation.note_state_matrix_to_midi(songs, "final")

class MidiManipulation:
    """
    DOCSTRING
    """
    def __init__(self):
        self.lower_bound = 24
        self.upper_bound = 102
        self.span = self.upper_bound - self.lower_bound

    def midi_to_note_state_matrix(midi_file, squash=True):
        """
        DOCSTRING
        """
        pattern = midi.read_midifile(midi_file)
        timeleft = [track[0].tick for track in pattern]
        posns = [0 for track in pattern]
        state_matrix, time = list(), 0
        state = [[0, 0] for x in range(self.span)]
        state_matrix.append(state)
        condition = True
        while condition:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(self.span)]
                state_matrix.append(state)
            for i in range(len(timeleft)):
                if not condition:
                    break
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]
                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < self.lower_bound) or (evt.pitch >= self.upper_bound):
                            pass
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-self.lower_bound] = [0, 0]
                            else:
                                state[evt.pitch - self.lower_bound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            out =  state_matrix
                            condition = False
                            break
                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None
                if timeleft[i] is not None:
                    timeleft[i] -= 1
            if all(t is None for t in timeleft):
                break
            time += 1
        S = numpy.array(state_matrix)
        state_matrix = numpy.hstack((S[:, :, 0], S[:, :, 1]))
        state_matrix = numpy.asarray(state_matrix).tolist()
        return state_matrix

    def note_state_matrix_to_midi(self, state_matrix, name="example"):
        """
        DOCSTRING
        """
        state_matrix = numpy.array(state_matrix)
        if not len(state_matrix.shape) == 3:
            state_matrix = numpy.dstack(
                (state_matrix[:,:self.span], state_matrix[:, self.span:]))
        state_matrix = numpy.asarray(state_matrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)
        tickscale, lastcmdtime = 55, 0
        prevstate = [[0, 0] for x in range(self.span)]
        for time, state in enumerate(state_matrix + [prevstate[:]]):
            offNotes, onNotes = list(), list()
            for i in range(self.span):
                n, p = state[i], prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(
                    tick=(time - lastcmdtime) * tickscale, pitch=note + lowerBound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(
                    tick=(time - lastcmdtime) * tickscale, velocity=40, pitch=note + lowerBound))
                lastcmdtime = time
            prevstate = state    
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
        midi.write_midifile("{}.mid".format(name), pattern)

class RBM_Chords:
    """
    DOCSTRING
    """
    def __init__(self):
        songs = get_songs('Pop_Music_Midi')
        print("{} songs processed".format(len(songs)))
        lowest_note = MidiManipulation.lower_bound
        highest_note = MidiManipulation.upper_bound
        note_range = highest_note-lowest_note
        num_timesteps = 15
        n_visible = 2 * note_range * num_timesteps
        n_hidden = 50
        num_epochs = 200
        batch_size = 100
        lr = tensorflow.constant(0.005, tensorflow.float32)
        x = tensorflow.placeholder(
            tensorflow.float32, [None, n_visible], name="x")
        W = tensorflow.Variable(
            tensorflow.random_normal([n_visible, n_hidden], 0.01), name="W")
        bh = tensorflow.Variable(
            tensorflow.zeros([1, n_hidden],  tensorflow.float32, name="bh"))
        bv = tensorflow.Variable(
            tensorflow.zeros([1, n_visible],  tensorflow.float32, name="bv"))
    
    def __call__(self):
        x_sample = gibbs_sample(1)
        h = sample(tensorflow.sigmoid(tensorflow.matmul(x, W) + bh))
        h_sample = sample(tensorflow.sigmoid(tensorflow.matmul(x_sample, W) + bh))
        size_bt = tensorflow.cast(tensorflow.shape(x)[0], tensorflow.float32)
        W_adder = tensorflow.multiply(lr/size_bt, tensorflow.subtract(
            tensorflow.matmul(tensorflow.transpose(x), h),
            tensorflow.matmul(tensorflow.transpose(x_sample), h_sample)))
        bv_adder = tensorflow.multiply(lr / size_bt, tensorflow.reduce_sum(
            tensorflow.subtract(x, x_sample), 0, True))
        bh_adder = tensorflow.multiply(lr / size_bt, tensorflow.reduce_sum(
            tensorflow.subtract(h, h_sample), 0, True))
        updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]
        with tensorflow.Session() as sess:
            init = tensorflow.global_variables_initializer()
            sess.run(init)
            for epoch in tqdm.tqdm(range(num_epochs)):
                for song in songs:
                    song = numpy.array(song)
                    song = song[:int(numpy.floor(
                        song.shape[0] / num_timesteps) * num_timesteps)]
                    song = numpy.reshape(
                        song, [song.shape[0] / num_timesteps, song.shape[1] * num_timesteps])
                    for i in range(1, len(song), batch_size): 
                        tr_x = song[i:i+batch_size]
                        sess.run(updt, feed_dict={x: tr_x})
            sample = gibbs_sample(1).eval(
                session=sess, feed_dict={x: numpy.zeros((50, n_visible))})
            for i in range(sample.shape[0]):
                if not any(sample[i, :]):
                    continue
                S = numpy.reshape(sample[i,:], (num_timesteps, 2*note_range))
                MidiManipulation.note_state_matrix_to_midi(S, "generated_chord_{}".format(i))
        
    def _gibbs_step(self, count, k, xk):
        """
        Runs a k-step gibbs chain to sample from the probability
        distribution of the RBM defined by W, bh, bv.
        """
        hk = sample(tensorflow.sigmoid(tensorflow.matmul(xk, W) + bh))
        xk = sample(tensorflow.sigmoid(tensorflow.matmul(
            hk, tensorflow.transpose(W)) + bv))
        return count + 1, k, xk

    def get_songs(self, path):
        """
        DOCSTRING
        """
        files = glob.glob('{}/*.mid*'.format(path))
        songs = list()
        for f in tqdm.tqdm(files):
            try:
                song = numpy.array(MidiManipulation.midi_to_note_state_matrix(f))
                if numpy.array(song).shape[0] > 50:
                    songs.append(song)
            except Exception as e:
                raise e
        return songs

    def gibbs_sample(self, k):
        """
        This function runs the gibbs chain.
        """
        ct = tensorflow.constant(0)
        [_, _, x_sample] = tensorflow.python.ops.control_flow_ops.while_loop(
            lambda count, num_iter, *args: count < num_iter,
            self._gibbs_step, [ct, tensorflow.constant(k), x])
        x_sample = tensorflow.stop_gradient(x_sample) 
        return x_sample

    def sample(self, probs):
        """
        This function lets us easily sample from a vector of probabilities 
        Takes in a vector of probabilitie.
        Returns a random vector of 0s and 1s sampled from the input vector.
        """
        return tensorflow.floor(probs + tensorflow.random_uniform(
            tensorflow.shape(probs), 0, 1))
