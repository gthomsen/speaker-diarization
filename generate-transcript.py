#!/usr/bin/env python3

# generates a transcription for the supplied media file's audio and writes the
# speaker-level, timestamped transcription to a Markdown table.

from __future__ import annotations

import copy
import getopt
import sys
from typing import ClassVar, Dict, IO, List, Tuple, Union

# keys in the whisper_model_details' dictionaries.
MODEL_NAME: str        = "model_name"
NUMBER_PARAMETERS: str = "number_parameters"
REQUIRED_VRAM: str     = "required_vram"
RELATIVE_SPEED: str    = "relative_speed"

# devices used for inference.
COMPUTE_DEVICE_CPU: str  = "cpu"
COMPUTE_DEVICE_CUDA: str = "cuda"

# data type used for the Whisper model's inference.  these are the data types
# supported by Faster Whisper.
COMPUTE_PRECISIONS: List[str] = [
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32"
]

# let the inference engine determine the best data type.
DEFAULT_COMPUTE_PRECISION = "auto"

# name of potential Whisper models to use for transcription.  these are
# enumerated as a convenience so users don't have to be experts on Whisper
# itself to use this script.  see the following link for additional details:
#
#   https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages
#
WHISPER_MODELS_DETAILS = [
    {
        MODEL_NAME:        "tiny",
        NUMBER_PARAMETERS: "39 M",
        REQUIRED_VRAM:     "<1 GiB",
        RELATIVE_SPEED:    "32x"
    },
    {
        MODEL_NAME:        "tiny",
        NUMBER_PARAMETERS: "74 M",
        REQUIRED_VRAM:     "1 GiB",
        RELATIVE_SPEED:    "16x"
    },
    {
        MODEL_NAME:        "small",
        NUMBER_PARAMETERS: "244 M",
        REQUIRED_VRAM:     "2 GiB",
        RELATIVE_SPEED:    "6x"
    },
    {
        MODEL_NAME:        "medium",
        NUMBER_PARAMETERS: "769 M",
        REQUIRED_VRAM:     "5 GiB",
        RELATIVE_SPEED:    "2x"
    },
    {
        # as of late 2023 large-v2 outperformed both large, large-v1 and
        # large-v3.
        MODEL_NAME:        "large-v2",
        NUMBER_PARAMETERS: "1550 M",
        REQUIRED_VRAM:     "10 GiB",
        RELATIVE_SPEED:    "1x"
    }
]

# name of the Whisper model to use for transcription.  we default to the
# largest and most performant model.
DEFAULT_WHISPER_MODEL: str = "large-v2"

# let WhisperX determine how many threads to use.
DEFAULT_NUMBER_THREADS: int = 0

# default to the maximum number of speakers available using a two digit
# enumeration scheme (pyannote names speakers as "SPEAKER_NN").
DEFAULT_NUMBER_SPEAKERS_RANGE = (1, 99)

class Options( object ):
    # boolean flag specifying whether transcription should be performed on the
    # CPU instead of a GPU.
    cpu_flag:   ClassVar[bool]

    # name of the WhisperX model to use for transcription.
    model_name: ClassVar[str]

    # number of speakers in the transcribed media as an inclusive range of
    # [minimum, maximum].
    number_speakers: ClassVar[Tuple[int, int]]

    # number of threads to use during transcription.  may be zero to let the
    # transcription library figure out how many threads to use.
    number_threads: ClassVar[int]

    # data type to use for inference.
    compute_data_type: ClassVar[str]

class Arguments( object ):
    # path to the Hugging Face access token for pyannote.
    access_token_path:  ClassVar[str]

    # path to the media file to transcribe.
    media_path:         ClassVar[str]

    # path to the file to write the transcription to.
    transcription_path: ClassVar[str]

def print_usage( program_name: str, file_handle: IO[str] = sys.stdout ) -> None:
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    def make_model_details_str() -> str:
        """
        Generates a string containing a table of the Whisper models available,
        along with basic details about their performance, suitable for printing
        in a help message.

        Takes no arguments.

        Returns 1 value:

          model_details_str - String containing the model details table.

        """

        # column names for the table.
        COLUMN_NAME       = "Name"
        COLUMN_PARAMETERS = "# Params"
        COLUMN_VRAM       = "VRAM (GiB)"
        COLUMN_SPEED      = "Relative Speed"

        # determine the width of each column.  we include the column's header
        # so everything is lined up.
        longest_name       = max( max( map( lambda s: len( s[MODEL_NAME] ),        WHISPER_MODELS_DETAILS ) ),
                                  len( COLUMN_NAME ) )
        longest_parameters = max( max( map( lambda s: len( s[NUMBER_PARAMETERS] ), WHISPER_MODELS_DETAILS ) ),
                                  len( COLUMN_PARAMETERS ) )
        longest_vram       = max( max( map( lambda s: len( s[REQUIRED_VRAM] ),     WHISPER_MODELS_DETAILS ) ),
                                  len( COLUMN_VRAM ) )
        longest_speed      = max( max( map( lambda s: len( s[RELATIVE_SPEED] ),    WHISPER_MODELS_DETAILS ) ),
                                  len( COLUMN_SPEED ) )

        # template for a row of details output.  each row is indented, the
        # names column left justified, the parameters column right justified,
        # and the VRAM and speed columns centered.  note the trailing newline.
        model_details_template = ("    " +
                                  "{name:<{name_width}s}    " +
                                  "{parameters:>{parameters_width}s}    " +
                                  "{vram:^{vram_width}s}    "
                                  "{speed:^{speed_width}s}\n")

        # start off with the column headers.
        model_details_str = model_details_template.format(
            name=COLUMN_NAME,             name_width=longest_name,
            parameters=COLUMN_PARAMETERS, parameters_width=longest_parameters,
            vram=COLUMN_VRAM,             vram_width=longest_vram,
            speed=COLUMN_SPEED,           speed_width=longest_speed )

        # add each of the models' details.
        for model_details in WHISPER_MODELS_DETAILS:
            model_details_str += model_details_template.format(
                name=model_details[MODEL_NAME],              name_width=longest_name,
                parameters=model_details[NUMBER_PARAMETERS], parameters_width=longest_parameters,
                vram=model_details[REQUIRED_VRAM],           vram_width=longest_vram,
                speed=model_details[RELATIVE_SPEED],         speed_width=longest_speed )

        # omit the trailing newline from the last entry in the table to simplify
        # the caller's formatting.
        return model_details_str[:-1]

    usage_str = """{program_name:s} [-C] [-d <compute_data_type>] [-h] [-m <model_name>] [-n <number_speakers>] [-t <number_threads>]<token_path> <media_path> <transcription_path>

Transcribes the audio contained in <media_path> and formats the speaker-level
segments into a Markdown table at <transcription_path>, overwriting it if it
already exists.  WhisperX is used to generate word-level timestamped segments
that are coalesced into sentence segments attributed to individual speakers
using OpenAI's Whisper model for transcription and pyannote-audio for speaker
diarization.

WhisperX uses ffmpeg to extract the audio from <media_path> allowing it to be
any file type that the ffmpeg found on the PATH can process (e.g. WAV, MP4, MKV,
M4A, MP3, etc).

By default, the {default_model_name:s} model is used for maximum performance albeit
requiring a non-trivial amount of GPU RAM to do so.  Other models available
include:

{model_details:s}

pyannote-audio (https://github.com/pyannote/pyannote-audio) requires users to
accept user conditions and generate a Hugging Face access token to use the
package.  Generate an access token by following these instructions:

  https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#tldr

Create a single line file at <token_path> containing only the token and give it
restrictive file permissions (e.g. 'chmod 600 access.token').

The command line options shown above are described below:

    -C                      Perform transcription with the host's CPU instead of an
                            available GPU.  If omitted, defaults to a GPU selected by
                            WhisperX.
    -d <compute_data_type>  Perform transcription using <compute_data_type> as the
                            inference data type.  Must be one of the following: {compute_data_types:s}.
                            Note that not all data types are available for each CPU
                            and GPU.
    -h                      Print this help message and exit.
    -m <model_name>         Use the Whisper model named <model_name>.  If omitted,
                            defaults to "{default_model_name:s}".
    -n <number_speakers>    Diarize the audio with <number_speakers>-many speakers.
                            <number_speakers> may be of the form <min_speakers:max_speakers>
                            to provide a range if the exact number is unknown.  If
                            omitted, defaults to "{min_number_speakers:d}:{max_number_speakers}".
    -t <number_threads>     Transcribe the audio using <number_threads>.  May be specified
                            as zero to let WhisperX select the number of threads to use.
                            If omitted, defaults to {default_number_threads:d} threads.  Note
                            that this has limited impact when using non-CPU
                            devices for inference.
""".format( compute_data_types=", ".join( map( lambda s: "'" + s + "'", COMPUTE_PRECISIONS ) ),
            default_model_name=DEFAULT_WHISPER_MODEL,
            default_number_threads=DEFAULT_NUMBER_THREADS,
            model_details=make_model_details_str(),
            min_number_speakers=DEFAULT_NUMBER_SPEAKERS_RANGE[0],
            max_number_speakers=DEFAULT_NUMBER_SPEAKERS_RANGE[1],
            program_name=program_name )

    print( usage_str, file=file_handle )

def fractional_seconds_to_hms( fractional_seconds: Union[int, float] ) -> Tuple(int, int, int):
    """
    Takes a fractional number of seconds and returns a tuple of the integral
    hours, minutes, and seconds it represents.

    Takes 1 argument:

      fractional_seconds - Number of seconds to convert.  May be any numeric type.

    Returns 3 values:

      hours   - Integral number of hours in fractional_seconds.
      minutes - Integral number of minutes in fractional_seconds.
      seconds - Integral number of seconds in fractional_seconds.

    """

    hours   = int( fractional_seconds / 3600 )
    minutes = int( (fractional_seconds - (hours * 3600)) / 60 )
    seconds = int( fractional_seconds ) % 60

    return hours, minutes, seconds

def fractional_seconds_to_hms_str( fractional_seconds: Union[int, float] ) -> str:
    """
    Takes a fractional number of seconds and returns a string of the form
    HH:MM:SS.

    Takes 1 argument:

      fractional_seconds - Number of seconds to convert.  May be any numeric type.

    Returns 1 value:

      fractional_seconds_str - String containing the "HH:MM:SS" representation of
                               fractional_seconds.

    """

    hours, minutes, seconds = fractional_seconds_to_hms( fractional_seconds )

    return "{:02d}:{:02d}:{:02d}".format( hours, minutes, seconds )

def get_access_token( access_token_path: str ) -> str:
    """
    Parses a file and returns the access token contained in it.  The path
    provided will be parsed as a text file and assumes the first non-empty line
    contains the token of interest.

    Takes 1 argument:

      access_token_path - Path to the file containing the access token.

    Returns 1 value:

      access_token - Access token string.

    """

    with open( access_token_path, "r" ) as access_token_fp:
        file_lines = access_token_fp.readlines()

        for file_line in file_lines:
            file_line = file_line.strip()
            if len( file_line ) == 0:
                continue

            return file_line

        raise ValueError( "No access token found!" )

def transcribe_and_diarize( audio_path: str,
                            model_name: str,
                            access_token: str,
                            transcription_device: str,
                            compute_data_type: str,
                            number_threads: int,
                            min_number_speakers: int = 1,
                            max_number_speakers: int = 99) -> Dict[List, List]:
    """
    Transcribes and diarizes the audio contained in the file path supplied using
    a Whisper model.  The caller specifies the acceleration device used, if any,
    for and the data type used for transcription.  The number of speakers can
    be provided to improve the diarization performance, if known.

    Takes 8 arguments:

      audio_path           - Path to the media file whose audio should be
                             transcribed.
      model_name           - Name of the Whisper model to use for transcription.
                             Must be one of the models found in WHISPER_MODELS_DETAILS.
      access_token         - Hugging Face access token to access the pyannote
                             diarization model.
      transcription_device - Compute device string specifying the device used
                             for transcription.  This must be supported by Faster
                             Whisper.
      compute_data_type    - String specifying the data type to use during
                             transcription.  This must be supported by Faster
                             Whisper as well as the transcription_device specified.
      number_threads       - Number of CPU threads to use during transcription.
                             May be specified as 0 to let WhisperX select the
                             number of threads.
      min_number_speakers  - Optional minimum number of speakers found in the audio
                             stream.  If omitted, defaults to 1.
      max_number_speakers  - Optional maximum number of speakers found in the audio
                             stream.  If omitted, defaults to 99.

    Returns 1 value:

      assigned_segments - Dictionary containing the speaker-assigned transcription
                          segments.  Contains "segments" and "word_segments" keys,
                          each of which contain a list of annotated dictionaries
                          (see coalesc_phrase_segments()).  The "segments" list
                          contains annotated phrases while "word_segments" contains
                          individually annotated words.

    """

    # load the workhorse package now that we know we're executing and not
    # just displaying a help message.
    import whisperx

    # load the model pipelines onto the target device.  note that this may take
    # a while if the model(s) need to be downloaded during the first execution.
    model_pipeline = whisperx.load_model( model_name,
                                          transcription_device,
                                          compute_type=compute_data_type,
                                          threads=number_threads )

    diarize_pipeline = whisperx.DiarizationPipeline( use_auth_token=access_token,
                                                     device=transcription_device )

    # pull the audio out of the media file.
    audio = whisperx.load_audio( audio_path )

    # transcribe and align the audio.
    transcription_result = model_pipeline.transcribe( audio, batch_size=16 )
    (alignment_pipeline,
     alignment_metadata) = whisperx.load_align_model( language_code=transcription_result["language"],
                                                      device=transcription_device )
    aligned_segments     = whisperx.align( transcription_result["segments"],
                                           alignment_pipeline,
                                           alignment_metadata,
                                           audio,
                                           transcription_device,
                                           return_char_alignments=False )

    # map the identified words to speakers.
    diarized_df       = diarize_pipeline( audio,
                                          min_speakers=min_number_speakers,
                                          max_speakers=max_number_speakers )
    assigned_segments = whisperx.assign_word_speakers( diarized_df,
                                                       aligned_segments )

    return assigned_segments

def coalesce_phrase_segments( phrase_segments: List ) -> List:
    """
    Coalesces a list of timestamped phrase segments into a list of timestamped
    speaker segments.  Consecutive phrase segments by the same speaker are
    coalesced into the same speaker segment so it is easier to track each side
    of a conversation.

    Word and sentence segments are described by dictionaries with the following
    structure:

      "words"   - Ignored.
      "speaker" - Speaker identifier for the segment.
      "text"    - Text contained in the segment.
      "start"   - Offset, in fractional seconds, from the beginning of the
                  transcription to the segment.
      "end"     - Offset, in fractional seconds, from the beginning of the
                  transcription to the end of the segment.

    Takes 1 argument:

      phrase_segments - List of dictionaries describing timestamped phrases.  See
                        above for the required keys and values.

    Returns 1 value:

      sentence_segments - List of dictionaries describing timestamped sentences.  See
                          above for keys and values.

    """

    #
    # NOTE: this has wonky logic for tracking the current speaker through
    #       word_segment["speaker"] which is hard to follow.  this should
    #       be cleaned up but it works well enough for now.
    #

    # coalesced set of segments.  each entry is a segment dictionary.
    sentence_segments = []

    # a single segment dictionary and the current speaker identifier.
    # the None speaker means we haven't started
    coalesced_segment = {}
    current_speaker   = None

    for phrase_segment in phrase_segments:
        # skip empty segments.
        if len( phrase_segment ) == 1 and "words" in phrase_segment:
            continue

        # make a copy so we can update this segment without altering what the
        # caller sees.
        phrase_segment = copy.deepcopy( phrase_segment )

        # assign segments to the current speaker if they don't have one for some
        # reason.  this is likely wrong, but we need a speaker to continue.
        phrase_segment["speaker"] = phrase_segment.get( "speaker", current_speaker )

        # have we changed speakers?  note that this also handles the
        # initialization of the first segment seen.
        if current_speaker is None or current_speaker != phrase_segment["speaker"]:
            # add the previous segment to the running sequence when we're not
            # starting up.
            if current_speaker is not None:
                sentence_segments.append( coalesced_segment )

            # track the current segment.
            current_speaker   = phrase_segment["speaker"]
            coalesced_segment = copy.deepcopy( phrase_segment )
            continue

        # add this segment to the current segment when the previous speaker is
        # still talking.
        elif current_speaker == phrase_segment["speaker"]:
            # add this segment's content to the coalesced segment.  extend the
            # coalesced segment to the end of the new segment.
            coalesced_segment["text"] += "  " + phrase_segment["text"]
            coalesced_segment["end"]   = phrase_segment["end"]

    # add the last segment into the list and reset our state.
    sentence_segments.append( coalesced_segment )

    return sentence_segments

def write_markdown_table( sentence_segments: List[Dict], markdown_path: str ):
    """
    Takes a list of sentence segments and writes them to file as a Markdown
    table.

    Sentence segments are described by dictionaries with the following structure:

      "speaker" - Speaker identifier for the segment.
      "text"    - Text contained in the segment.
      "start"   - Offset, in fractional seconds, from the beginning of the
                  transcription to the segment.
      "end"     - Offset, in fractional seconds, from the beginning of the
                  transcription to the end of the segment.

    Takes 2 arguments:

      sentence_segments - List of sentence segments to write as a table.
      markdown_path     - Path to the file to write the transcription table.  If
                          markdown_path already exists it is overwritten.

    Returns nothing.

    """

    table_header = "| Time Frame | Speaker | Words Spoken |\n| --- | --- | --- |"

    with open( markdown_path, "w" ) as markdown_fp:
        # write out the table header.
        print( table_header,
               file=markdown_fp )

        # write each speaker's segment as a new row in our table.
        for sentence_segment in sentence_segments:
            start_time_str = fractional_seconds_to_hms_str( sentence_segment["start"] )
            end_time_str   = fractional_seconds_to_hms_str( sentence_segment["end"] )

            print( "| {:s}-{:s} | {:s} | {:s} |".format(
                start_time_str,
                end_time_str,
                sentence_segment["speaker"],
                sentence_segment["text"] ),
                   file=markdown_fp )

        # end the file with an empty line.
        print( "", file=markdown_fp )

def parse_command_line( argv: List[str] ) -> Union[Tuple[Options, Arguments], Tuple[None, None]]:
    """
    Parses the script's command line into two objects whose attributes contain
    the script's execution parameters.

    Takes 1 argument:

      argv - List of strings representing the command line to parse.  Assumes the
             first string is the name of the script executing.

    Returns 2 values:

      options   - Options object whose attributes represent the optional flags parsed.

                  NOTE: Will be None if execution is not required.

      arguments - Arguments object whose attributes represent the positional arguments
                  parsed.

                  NOTE: Will be None if execution is not required.

    """

    # indicies into argv's positional arguments specifying each of the required
    # arguments.
    ARG_ACCESS_TOKEN_PATH    = 0
    ARG_MEDIA_PATH           = 1
    ARG_TRANSCRIPTION_PATH   = 2
    NUMBER_MINIMUM_ARGUMENTS = ARG_TRANSCRIPTION_PATH + 1

    options, arguments = Options(), Arguments()

    # set the defaults for the options.

    # default to GPU acceleration with our default model.
    options.cpu_flag          = False
    options.model_name        = DEFAULT_WHISPER_MODEL
    options.number_speakers   = DEFAULT_NUMBER_SPEAKERS_RANGE
    options.compute_data_type = DEFAULT_COMPUTE_PRECISION
    options.number_threads    = DEFAULT_NUMBER_THREADS

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "Cd:hm:n:t:")
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-C":
            options.cpu_flag = True
        if option == "-d":
            if option_value.lower() not in COMPUTE_PRECISIONS:
                raise ValueError( "Unknown compute precision specified ({:s})!".format(
                    option_value ) )
            options.compute_data_type = option_value.lower()
        elif option == "-h":
            print_usage( argv[0] )
            return (None, None)
        elif option == "-m":
            options.model_name = option_value
        elif option == "-n":
            try:
                options.number_speakers = tuple( map( lambda n: int( n ),
                                                      option_value.split( ":", 1 ) ) )

                # treat a single speaker limit as both the lower- and upper
                # bound.
                if len( options.number_speakers ) == 1:
                    options.number_speakers = (options.number_speakers[0],
                                               options.number_speakers[0])
            except ValueError:
                raise ValueError( "Number of speakers must be specified as N or N:M ({:s})".format(
                    option_value ) )
        elif option == "-t":
            try:
                options.number_threads = int( option_value )
            except ValueError:
                raise ValueError( "Invalid number of threads specified ({:s})".format(
                    option_value ) )

    # ensure we have enough positional arguments.
    if len( positional_arguments ) < NUMBER_MINIMUM_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected at least {:d} but "
                          "received {:d}.".format(
                              NUMBER_MINIMUM_ARGUMENTS,
                              len( positional_arguments ) ) )

    # ensure the range of speakers expected is sensible.
    if not ((1 <= options.number_speakers[0]) and
            (options.number_speakers[0] <= options.number_speakers[1]) and
            (options.number_speakers[1] <= DEFAULT_NUMBER_SPEAKERS_RANGE[1])):
        raise ValueError( "Number of speakers must be a subset of the range [1, 99] ({:d}, {:d})".format(
            *options.number_speakers ) )

    # ensure the number of threads is reasonable.
    if options.number_threads < 0:
        raise ValueError( "Number of threads must be non-negative ({:d})".format(
            options.number_threads ) )

    # map the positional arguments to named variables.
    arguments.access_token_path  = positional_arguments[ARG_ACCESS_TOKEN_PATH]
    arguments.media_path         = positional_arguments[ARG_MEDIA_PATH]
    arguments.transcription_path = positional_arguments[ARG_TRANSCRIPTION_PATH]

    return options, arguments

def main( argv: List[str] ):
    """
    Driver for the script.  Verifies the arguments supplied on the script's
    command line, extract an audio stream, and transcribes it to Markdown.

    Takes 1 argument:

      argv - Sequence of strings specifying the script's command line, starting with the
             script's path.

    Returns 1 value

      exit_status - Integer status code to exit the script with.

    """

    try:
        options, arguments = parse_command_line( argv )
    except Exception as e:
        print( str( e ), file=sys.stderr )
        return 1

    # return success in the case where normal execution is not required, but
    # we're not in an exceptional situation (e.g. requested help).
    if (options is None) and (arguments is None):
        return 0

    # use the GPU unless the host was explicitly requested.
    #
    # NOTE: we explicitly structure the logic so that it is easy to add support
    #       for additional devices.
    #
    # NOTE: as of 2024/07/08 WhisperX uses Faster Whisper and CTranslate2 for
    #       inference, neither of which supports Apple Silicon's GPU (via Metal
    #       Performance Shaders).  CTranslate2 4.3.1 and faster-whisper 1.0.3
    #       only support CPU and CUDA for inference.
    #
    import torch
    if options.cpu_flag:
        transcription_device = COMPUTE_DEVICE_CPU
    elif torch.cuda.is_available():
        transcription_device = COMPUTE_DEVICE_CUDA
    else:
        # doesn't look like a GPU is available so fall back to the CPU.
        transcription_device = COMPUTE_DEVICE_CPU

    # get our Hugging Face access token.
    access_token = get_access_token( arguments.access_token_path )

    # load the audio, model and transcribe into word-level segments.
    phrase_segments = transcribe_and_diarize( arguments.media_path,
                                              options.model_name,
                                              access_token,
                                              transcription_device,
                                              options.compute_data_type,
                                              options.number_threads,
                                              min_number_speakers=options.number_speakers[0],
                                              max_number_speakers=options.number_speakers[1] )

    # coalesce word-level segments to speaker-level segments.
    sentence_segments = coalesce_phrase_segments( phrase_segments["segments"] )

    # write out the transcript in Markdown.
    write_markdown_table( sentence_segments, arguments.transcription_path )

    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
