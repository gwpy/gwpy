#!/usr/bin/env python

"""Format changelog
"""

import argparse
import datetime
import os.path
import platform

import git

if platform.linux_distribution()[0] == 'debian':
    DEFAULT_FORMAT = 'deb'
else:
    DEFAULT_FORMAT = 'rpm'


def format_rpm(tag):
    tago = tag.tag
    date = datetime.date.fromtimestamp(tago.tagged_date)
    dstr = date.strftime('%a %b %d %Y')
    tagger = tago.tagger
    message = tago.message.split('\n')[0]
    return "* {} {} <{}>\n- {}\n".format(dstr, tagger.name, tagger.email,
                                         message)


def format_deb(tag):
    tago = tag.tag
    date = datetime.datetime.fromtimestamp(tago.tagged_date)
    dstr = date.strftime('%a, %d %b %Y %H:%M:%D')
    tz = int(tago.tagger_tz_offset / 3600. * 100)
    version = tag.name.strip('v')
    tagger = tago.tagger
    message = tago.message.split('\n')[0]
    return (" -- {} <{}>  {} {}\n\n"
            "gwpy ({}-1) unstable; urgency=low\n\n"
            "  * {}\n".format(tagger.name, tagger.email, dstr, tz,
                              version, message))


formatter = {
    'rpm': format_rpm,
    'deb': format_deb,
}

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-f', '--changelog-format', default=DEFAULT_FORMAT,
                    choices=list(formatter.keys()),
                    help='output changelog format')
parser.add_argument('-r', '--repo-directory', default=os.path.curdir,
                    help='path of git repository')
parser.add_argument('-s', '--start-tag', default=None,
                    help='tag to start with')
args = parser.parse_args()

repo = git.Repo(args.repo_directory)
tags = repo.tags
tags.reverse()

func = formatter[args.changelog_format]

if args.start_tag:
    args.start_tag = args.start_tag.lstrip('v')

for tag in tags:
    if args.start_tag and tag.name.lstrip('v') < args.start_tag:
        break
    try:
        print(func(tag))
    except AttributeError:
        continue
