#!/bin/bash
export AWS_SECRET_ACCESS_KEY=$(cat .env.local | grep AWS_SECRET | cut -d "\"" -f 2)
export AWS_ACCESS_KEY=$(cat .env.local | grep AWS_ACCESS | cut -d "\"" -f 2)
aws s3 ls models.webaverse.com | grep glb | cut -d " " -f 6 | cut -d "." -f 1


