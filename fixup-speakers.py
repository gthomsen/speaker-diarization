#!/usr/bin/env python3

# remaps speaker names in a transcription Markdown table.

import getopt
import sys
from typing import ClassVar, Dict, IO, List, Tuple, Union

# match the speaker names produced by pyannote.
SPEAKER_NAME_TEMPLATE: str = "SPEAKER_{:02d}"

class Options( object ):
    # there are no options for this script.

    pass

class Arguments( object ):
    # path to the transcription to update.
    input_path:  ClassVar[str]

    # path to the updated transcription.
    output_path: ClassVar[str]

    # dictionary mapping original speaker names to new speaker names.
    speaker_map: ClassVar[Dict[str, str]]

    pass

def print_usage( program_name: str, file_handle: IO[str] = sys.stdout ) -> None:
    """
    Prints the script's usage to standard output.

    Takes 2 arguments:

      program_name - Name of the program currently executing.
      file_handle  - File handle to print to.  If omitted, defaults to standard output.

    Returns nothing.

    """

    usage_str = """{program_name:s} [-h] <input_path> <output_path> <speaker1>[:<new_speaker1>] [...]

Remaps speaker names in a Markdown table containing a transcription.  Parses the
table contained in <input_path>, replaces speaker names according to the
<speaker>:<new_speaker> mappings provided, and writes a new table to <output_path>.
If <output_path> already exists, it is overwritten.  <input_path> is fully parsed
before performing speaker remapping allowing <output_path> to be identical and
performing an in-place update.

The transcription table is assumed to have speaker names in the second column as
output by generate-transcript.py.

Speaker names containing spaces must be escaped some how, either by quoting the
entire <speaker> argument like "Bob Smith", or escaping each space like
Bob\ Smith.

The command line options shown above are described below:

    -h                  Print this help message and exit.
""".format( program_name=program_name )

    print( usage_str, file=file_handle )

def parse_speaker_map( speaker_map_list: str ) -> Dict[str, str]:
    """
    Generates a mapping from old to new speakers from a list of speakers.  Each
    entry may either be of the form "<new_speaker>" or
    "<old_speaker>:<new_speaker>".  In the case where only a new speaker is
    provided, it is assumed that the corresponding old speaker's name is of the
    form "SPEAKER_NN", where "NN" is the position in the provided list.

    Raises ValueError if no mappings are provided as well as if too many (more
    than 99) mappings are provided.

    Takes 1 argument:

      speaker_map_list - List of strings speaker mapping strings either of the
                         form of "<new_speaker>" or
                         "<old_speaker>:<new_speaker>".  One or both may be
                         present in the supplied list.

    Returns 1 value:

      speaker_map - Dictionary mapping old speakers to new speakers.

    """

    speaker_map = {}

    number_speakers = len( speaker_map_list )

    # ensure that we were called correctly.
    if number_speakers == 0:
        raise ValueError( "List of speakers provided must contain at least one speaker" )
    elif number_speakers > 99:
        # our default speaker format is "SPEAKER_NN" so as to match the output
        # of WhisperX.  complain when we cannot fit into NN.
        raise ValueError( "Must not have more than 100 speakers" )

    for speaker_number, speaker_map_str in enumerate( speaker_map_list, 1 ):
        components: List[str] = speaker_map_str.split( ":", 2 )

        # use the default speaker name if we weren't provided with an (old, new)
        # mapping.  use the position in the speaker map to fill out the template
        # instead of counting how many times the template was previously used.
        if len( components ) == 1:
            components = [SPEAKER_NAME_TEMPLATE.format( speaker_number ),
                          components[0]]

        if len( components[1].strip() ) == 0:
            raise ValueError( "Speaker #{:d} is an empty string! ({:s})".format(
                speaker_number,
                components[1] ) )

        # add this mapping.
        speaker_map[components[0]] = components[1]

    return speaker_map

def parse_markdown_table( markdown_path: str ) -> Tuple[str, List[str]]:
    """
    Parses a markdown table from the supplied path.  Lines prior to the Markdown
    table header, if any, are ignored.  All lines after the Markdown table
    header are assumed to be the table itself.

    Raises ValueError if the supplied path does not contain a table.

    Takes 1 argument:

      markdown_path - Path to the Markdown file to parse.

    Returns 2 arguments:

      table_header - String containing the table's header lines.
      table_rows   - List of strings containing the table's contents.

    """

    table_header = ""
    table_rows   = []

    with open( markdown_path, "r" ) as markdown_fp:
        markdown_lines = markdown_fp.readlines()

        for line_index, markdown_line in enumerate( markdown_lines ):

            # ignore anything before the first Markdown table and then assume
            # the rest of the file is the table.
            if markdown_line.strip().startswith( "|" ):
                table_header = "\n".join( markdown_lines[:line_index] )
                table_rows   = markdown_lines[line_index:]

                break
        else:
            raise ValueError( "'{:s}' did not contain a Markdown table!".format(
                markdown_path ) )

    return table_header, table_rows

def fixup_speakers( table_rows: List[str], speaker_map: Dict[str, str] ) -> List[str]:
    """
    Takes a list of Markdown table rows and a mapping of old to new speaker
    names, and updates the rows' speaker column according to the mapping.

    Takes 2 arguments:

      table_rows  - List of strings containing the rows of a Markdown table.
                    This may include the table's header.
      speaker_map - Dictionary mapping old speakers to new.  Speakers in
                    table_rows that are not in this map are left as is.

    Returns 1 value:

      updated_table_rows - List of strings containing the updated rows of
                           table_rows.

    """

    updated_table_rows = []

    # compute the length of the longest speaker name so we can align the
    # speakers' column.  this greatly improves readability of the table after
    # speakers are fixed up.
    maximum_speaker_name_length = max( len( SPEAKER_NAME_TEMPLATE ),
                                       max( map( lambda s: len( s ), speaker_map.values() ) ) )

    for table_row in table_rows:
        # break the table into columns.
        row_columns = table_row.split( "|" )

        # ignore non-table lines.
        if len( row_columns ) == 1:
            updated_table_rows.append( table_row )
            continue

        # we assume the 2nd column only holds the speaker name.  keep in mind
        # that tables start with a vertical bar, so the 2nd column is the third
        # entry in the list.
        speaker_name = row_columns[2].strip()

        # we have nothing to do if this is an unknown speaker.  this also
        # handles table headers and the header/content separation line.
        if speaker_name not in speaker_map:
            updated_table_rows.append( table_row )
            continue

        # pad the new speaker's name with a single space on either side.  we
        # pad each speaker so that it's aligned with the longest speakers name
        # in our map so that the raw Markdown is easier to read.
        #
        # NOTE: this does not preserve the original formatting, which we
        #       don't care enough to spend the time to implement.
        #
        row_columns[2] = " {:{:d}s} ".format( speaker_map[speaker_name],
                                              maximum_speaker_name_length )

        table_row = "|".join( row_columns )

        updated_table_rows.append( table_row )

    return updated_table_rows

def write_markdown_table( output_path: str, table_header: str, table_rows: List[str] ):
    """
    Writes a Markdown table to the supplied path.  The table header is supplied
    separately from its rows.

    Takes 3 arguments:

      output_path  - Path to write the Markdown table to.  If this exists it is
                     overwritten.
      table_header - Markdown table header string.  This is the first thing
                     written to output_path.
      table_rows   - List of Markdown table rows.

    Returns nothing.

    """

    # reconstruct our Markdown table.  both the header and rows have newlines so
    # we suppress adding additional new lines.
    with open( output_path, "w" ) as markdown_fp:
        print( table_header, file=markdown_fp, end="" )

        for table_row in table_rows:
            print( table_row, file=markdown_fp, end="" )

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
    ARG_INPUT_PATH  = 0
    ARG_OUTPUT_PATH = 1
    ARG_SPEAKER_MAP = 2
    NUMBER_MINIMUM_ARGUMENTS = ARG_SPEAKER_MAP + 1

    options, arguments = Options(), Arguments()

    # parse our command line options.
    try:
        option_flags, positional_arguments = getopt.getopt( argv[1:], "h")
    except getopt.GetoptError as error:
        raise ValueError( "Error processing option: {:s}\n".format( str( error ) ) )

    # handle any valid options that were presented.
    for option, option_value in option_flags:
        if option == "-h":
            print_usage( argv[0] )
            return (None, None)

    # ensure we have enough positional arguments.
    if len( positional_arguments ) < NUMBER_MINIMUM_ARGUMENTS:
        raise ValueError( "Incorrect number of arguments.  Expected at least {:d} but "
                          "received {:d}.".format(
                              NUMBER_MINIMUM_ARGUMENTS,
                              len( positional_arguments ) ) )

    # map the positional arguments to named variables.
    arguments.input_path  = positional_arguments[ARG_INPUT_PATH]
    arguments.output_path = positional_arguments[ARG_OUTPUT_PATH]

    try:
        arguments.speaker_map = parse_speaker_map( positional_arguments[ARG_SPEAKER_MAP:] )
    except ValueError as e:
        raise ValueError( "Failed to parse the speaker map - {:s}!".format( str( e ) ) )

    return options, arguments

def main( argv ):
    """
    Driver for the script.  Verifies the arguments supplied on the script's
    command line, parses the transcript, and remaps the speaker names present.

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

    table_header, table_rows = parse_markdown_table( arguments.input_path )

    table_rows = fixup_speakers( table_rows, arguments.speaker_map )

    write_markdown_table( arguments.output_path, table_header, table_rows )

    return 0

if __name__ == "__main__":
    sys.exit( main( sys.argv ) )
