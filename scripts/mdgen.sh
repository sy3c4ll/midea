#!/bin/bash
set -e
cd "$1"
find *|grep "\\.mid">"$2"
