#!/bin/bash
export AWS_SECRET_ACCESS_KEY=$(cat .env.local | grep AWS_SECRET | cut -d "\"" -f 2)
export AWS_ACCESS_KEY=$(cat .env.local | grep AWS_ACCESS | cut -d "\"" -f 2)
aws s3 cp model.glb 's3://models.webaverse.com/'$1'.glb'
rm model.glb

