# Overview

Simple tools to provide transcription of audio/video recordings using
[WhisperX](https://github.com/m-bain/whisperX) for word-level timestamping and
speaker diarization via OpenAI's Whisper model.  Transcripts are generated as
Markdown tables with each row containing a contiguous region of a single
speaker's speech.  This makes it easy read the text and find the corresponding
audio/video by timestamp, as well as further organize the transcription by
sections.

Transcript generation is separate from speaker identity assignment and handled
by separate scripts (`generate-transcript.py` and `fixup-speakers.py`,
respectively) so as to support a workflow that involves transcript review prior
to speaker assignment.  This results in commands like the following:

``` shell
$ ./generate-transcript.py \
    hf.token \
    meeting.mp4 \
    meeting-transcript-raw.md
$ ./fixup-speakers.py \
    meeting-transcript-raw.md \
    meeting-transcript.md \
    "Bob Smith" \
    "Nancy Jones" \
    "Moderator"
```

# Example Usage

By default GPU acceleration is used when available so nothing special is
required to enable it:

``` shell
$ ./generate-transcript.py \
    hf.token \
    meeting.mp4 \
    meeting-transcript-raw.md
```

Specifying CPU inference (`-C`) with a thread count (`-n <threads>`):

```shell
$ ./generate-transcript.py \
    -C -t 16 \
    hf.token \
    meeting.mp4 \
    meeting-transcript-raw.md
```

The number of speakers can be specified if known ahead of time:

```shell
# assume two speakers during transcription.
$ ./generate-transcript.py \
    -n 2 \
    hf.token \
    meeting.mp4 \
    meeting-transcript-raw.md
```

Ranges of speaker counts can be specified if a precise count isn't known:

```shell
# assume no more than five speakers during transcription.
$ ./generate-transcript.py \
    -n 1:5 \
    hf.token \
    meeting.mp4 \
    meeting-transcript-raw.md
```

The most capable transcription model (read: largest) is used by default, though
others may be used instead:

```shell
# use the "small.en" Whisper model.
$ ./generate-transcript.py \
    -m small.en \
    hf.token \
    meeting.mp4 \
    meeting-transcript-raw.md
```

See the
[Whisper repository](https://github.com/openai/whisper#available-models-and-languages) for
details on which models are available.

Transcription can be done with lower precision for faster execution.  Note that
not all data types are available for all types of CPUs and GPUs!

```shell
# perform transcription using 8-bit integers.
$ ./generate-transcript.py \
    -d int8 \
    hf.token \
    meeting.mp4 \
    meeting-transcript-raw.md
```

Each script provides help when executed with the `-h` option.  Please refer to
the help messages for additional details.

# Installation

This repository's tools relies on the following dependencies:

* Anaconda Python
* Python 3.10
* [`pyannote-audio`](https://github.com/pyannote/pyannote-audio)
* PyTorch 2.0.0 with CUDA 11.8 (optional but ***strongly*** recommended)

The use of `pyannote-audio` requires users to accept conditions on using its
sub-packages
([`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0), and
[`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1)),
as well as generating a Hugging Face access token to access the model
([https://hf.co/settings/tokens](https://hf.co/settings/tokens)).

Copy the Hugging Face access token to a file (and ensure it has restrictive
permissions, e.g. `chmod 600 hf.token`).

## Linux

GPU-accelerated inference using CUDA should be strongly preferred unless you
know what you're doing.

### CUDA Inference

***NOTE:*** We have to work around a packaging issue with Torch 2 (possibly
TorchVision >=1.14 as well).  Add the `export LD_LIBRARY_PATH=...` line into
your shell's configuration (e.g. `.bashrc`).  See this [Stack Overflow
post](https://stackoverflow.com/questions/75879951/torch-2-installed-could-not-load-library-libcudnn-cnn-infer-so-8-error-libnv)
for additional details.

```shell
$ conda create -n transcription python=3.10
$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install whisperx
$ export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/:${LD_LIBRARY_PATH}
$ ln -s ${CONDA_PREFIX}/lib/libnvrtc.so.11.8.* ${CONDA_PREFIX}/lib/libnvrtc.so
```

### CPU Inference

```shell
$ conda create -n transcription python=3.10
$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch
$ pip install whisperx
```

## MacOS

Note that as of 2024/07/08 `faster-whisper` 1.0.3, used by `whisper` for
inference, does not support acceleration via Metal Performance Shaders (MPS)
which means all transcription is performed on the CPU and not the GPU.

```shell
$ conda create -n transcription python=3.10
$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch
$ pip install whisperx
```

# Limitations

This is a quick and dirty solution and does not attempt to get perfect
transcripts.  Challenging areas where multiple speakers talk simultaneously
often generates a mixed transcript where Speaker B's text is attributed to
Speaker A.
