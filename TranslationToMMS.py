import re

import pandas

import xml.etree.ElementTree as ET


# Default: this is the average transition between glosses in seconds
DEFAULT_TRANSITION = 0.35

def _extract_text_from_xml(xml_tree) -> str:
    """Extracts the text between <value><string> tags from an XML file."""

    root = xml_tree.getroot()

    value_element = root.find('.//value')
    if value_element is not None:
      string_element = value_element.find('string')
      if string_element is not None:
        return string_element.text

    raise Exception("Elements <value><string> not found")


def _recapitalize_gloss(gloss: str) -> str:
    """The translator outputs everything in lo-case.
    This function brings the case up again where needed.
    Strategy:
    1) if there is a column (category specifier), capitalize only the part after the column;
    2) if in the second part there are parentheses, leave the content in parentheses in lo-case."""

    column_pos = gloss.find(':')
    if column_pos == -1:
        category = ""
        content = gloss
    else:
        category = gloss[:column_pos] + ":"
        content = gloss[column_pos + 1: ]


    # Look for parentheses
    par_match_res = re.match("(.+)(\(.+\))", content)
    if par_match_res:
        # We found parentheses
        out_par = par_match_res.group(1)
        in_par = par_match_res.group(2)

        # In general, up-case only the part outside the parentheses

        # Notable exceptions to the general rule!
        if in_par == "(-ort)":
            in_par = "(-Ort)"
        elif in_par == "(-objekt)":
            in_par = "(-Objekt)"

        up_cased = out_par.upper() + in_par
    else:
        up_cased = content.upper()

    out = category + up_cased

    return out

def format_mms(in_text: str, transition_duration: float = DEFAULT_TRANSITION) -> pandas.DataFrame:
    """Given the sequence of glosses, as single string, composes a dataframe containing an MMS instance
    with default duration and fixed transition time between glosses"""

    glosses = in_text.split()

    n_glosses = len(glosses)

    out_glosses = []

    for gloss in glosses:
        recap_gloss = _recapitalize_gloss(gloss)
        print(f"{gloss} --> {recap_gloss}")
        out_glosses.append(recap_gloss)

    assert len(out_glosses) == n_glosses

    mms_data = {
        'maingloss': out_glosses,
        'framestart': [0] * n_glosses,
        'frameend': [0] * n_glosses,
        'duration': ['100%'] * n_glosses,
        'transition': [transition_duration] * n_glosses,
        'domgloss': [''] * n_glosses,
        'ndomgloss': [''] * n_glosses,
    }

    out = pandas.DataFrame(data=mms_data)
    return out


#
#
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Converts the lo-case output of the translator into an MMS instance.')
    parser.add_argument('--in-xml', '-i', type=str,
                        help='The filename with the text to format into an MMS, as outputted by the translator.',
                        required=True)
    parser.add_argument('--out-mms', '-o', type=str,
                        help='The name of the file that will contain the MMS instance.',
                        required=True)

    args = parser.parse_args()

    in_xml_path = Path(args.in_xml)
    out_mms_path = Path(args.out_mms)

    if not in_xml_path.exists():
        raise Exception(f"File {str(in_xml_path)} doesn't exist.")

    print(f"Parsing XML file '{in_xml_path}'...")
    tree = ET.parse(in_xml_path)
    in_txt = _extract_text_from_xml(xml_tree=tree)

    print(f"Input text: '{in_txt}'")

    mms_dataframe: pandas.DataFrame = format_mms(in_text=in_txt)
    print(f"Output MMS size: {len(mms_dataframe)}")

    print(f"Saving output MMS to {out_mms_path}...")
    mms_dataframe.to_csv(path_or_buf=out_mms_path, sep=",", header=True, index=False)

    print("All Done.")
