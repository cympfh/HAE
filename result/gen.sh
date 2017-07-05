#!/bin/bash

for epoch in $(seq 0 1000); do
    if [ ! -f reconstructed.$epoch.0.gif ]; then
        break
    fi
    echo "<h1>Epoch $epoch</h1>"
    echo "<h2>reconstructed</h2>"
    for i in $(seq 0 9); do
        echo "<img src=reconstructed.$epoch.$i.gif>"
    done
    echo "<h2>fake reconstructed</h2>"
    for i in $(seq 0 9); do
        echo "<img src=fake-reconstructed.$epoch.$i.gif>"
    done
    echo "<h2>style-transfered generated</h2>"
    for i in $(seq 0 9); do
        echo "<img src=generated.$epoch.$i.gif>"
    done
done >index.html
