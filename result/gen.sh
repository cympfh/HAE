#!/bin/bash

iota() {
    seq 0 $(($1 - 1))
}

testing() {
    echo "<table>"
    echo "<tr align=right><th>original</th>"
    for i in `iota 10`; do
        echo "<td>${i}</td>"
    done
    echo "</tr>"

    for i in `iota 10`; do
        echo "<tr>"
        echo "<td><img src=original.${i}.gif></td>"
        for j in `iota 10`; do
            IMG="trans.${i}.${j}.gif"
            echo "<td><img src=$IMG></td>"
        done
        echo "</tr>"
    done
    echo "</table>"
}

learning() {
    for epoch in $(iota 10); do
        echo "<h1>Epoch $epoch</h1>"
        echo "<h2>reconstructed</h2>"
        for i in $(iota 10); do
            echo "<img src=reconstructed.$epoch.$i.gif>"
        done
        echo "<h2>fake reconstructed</h2>"
        for i in $(iota 10); do
            echo "<img src=fake-reconstructed.$epoch.$i.gif>"
        done
        echo "<h2>style-transfered generated</h2>"
        for i in $(iota 10); do
            echo "<img src=generated.$epoch.$i.gif>"
        done
    done
}

learning > index.html
testing >> index.html
