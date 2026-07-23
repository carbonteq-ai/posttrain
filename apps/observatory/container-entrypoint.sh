#!/bin/sh
set -eu

if [ "${POSTTRAIN_OBSERVATORY_SOURCE:-}" = "trackio" ]; then
    : "${POSTTRAIN_TRACKIO_PROJECT:?POSTTRAIN_TRACKIO_PROJECT is required for the Trackio source}"
    : "${POSTTRAIN_TRACKIO_SERVER_URL:?POSTTRAIN_TRACKIO_SERVER_URL is required for the containerized Trackio source}"
fi

exec posttrain-observatory "$@"
