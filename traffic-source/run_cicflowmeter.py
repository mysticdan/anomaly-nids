#!/usr/bin/env python3
"""Small wrapper around python-cicflowmeter CLI bugs."""

import argparse

from cicflowmeter.sniffer import create_sniffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--fields", required=True)
    args = parser.parse_args()

    sniffer, session = create_sniffer(
        input_file=None,
        input_interface=args.interface,
        output_mode="csv",
        output=args.output,
        fields=args.fields,
        input_directory=None,
    )
    sniffer.start()
    try:
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()
    finally:
        if hasattr(session, "_gc_stop"):
            session._gc_stop.set()
            session._gc_thread.join(timeout=2.0)
        session.flush_flows()


if __name__ == "__main__":
    main()
