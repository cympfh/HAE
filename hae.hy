(import
  [click]
  [json]
  [datetime]
  [numpy]
  [PIL [Image]]
  [keras.backend [tensorflow_backend :as K]]
  [keras.layers [Dense Dropout Reshape Flatten Input]]
  [keras.layers.convolutional [Conv2D UpSampling2D]]
  ; [keras.layers.normalization [BatchNormalization]]
  [keras.layers.pooling [MaxPooling2D]]
  [keras.layers.merge [concatenate]]
  [keras.optimizers [Adam]]
  [keras.models [Sequential Model]]
  [keras.datasets [mnist]]
  [keras.utils [np_utils]])

(def HIDDEN-DIM 64)


(defn load-data []
     (def [[x_train y_train] [x_test y_test]] (mnist.load_data))
     (setv x_train (/ (.astype x_train "f") 255))
     (setv x_test (/ (.astype x_test "f") 255))
     (setv y_train (np_utils.to_categorical y_train 10))
     (setv y_test (np_utils.to_categorical y_test 10))
     (, (, x_train y_train) (, x_test y_test)))


(defn build-encoder []
     "28x28 -> hidden-dim vector"
     (doto
       (Sequential :name 'encoder)
       (.add (Reshape (, 28 28 1) :input_shape (, 28 28)))
       (.add (Conv2D 32 (, 3 3) :padding 'same :activation 'relu))
       (.add (MaxPooling2D (, 2 2) :padding 'same))
       (.add (Conv2D 32 (, 3 3) :padding 'same :activation 'relu))
       (.add (MaxPooling2D (, 2 2) :padding 'same))
       (.add (Flatten))
       (.add (Dense HIDDEN-DIM))
       (.summary)))


(defn build-decoder []
      "{hidden-dim 10-dim} => 28x28"
      (def z (Input (, HIDDEN-DIM)))
      (def y (Input (, 10)))
      (def reconstructed-x
           (-> [z ((Dense 32 :activation 'relu) y)]
               concatenate
               ((Dense (* 7 7 32) :activation 'relu))
               ((Reshape (, 7 7 32)))
               ((Conv2D 32 (, 3 3) :padding 'same :activation 'relu))
               ((UpSampling2D :size (, 2 2)))
               ((Conv2D 32 (, 3 3) :padding 'same :activation 'relu))
               ((UpSampling2D :size (, 2 2)))
               ((Conv2D 1 (, 3 3) :padding 'same :activation 'sigmoid))
               ((Reshape (, 28 28)))))
      (doto
        (Model [z y] reconstructed-x)
        (.summary)))


(defn build-fake-decoder []
      "hidden-dim => 28x28"
      (doto
        (Sequential :name 'fake-decoder)
        (.add (Dense (* 7 7 32) :activation 'relu :input_shape (, HIDDEN-DIM)))
        (.add (Reshape (, 7 7 32)))
        (.add (Conv2D 32 (, 3 3) :padding 'same :activation 'relu))
        (.add (UpSampling2D :size (, 2 2)))
        (.add (Conv2D 32 (, 3 3) :padding 'same :activation 'relu))
        (.add (UpSampling2D :size (, 2 2)))
        (.add (Conv2D 1 (, 3 3) :padding 'same :activation 'sigmoid))
        (.add (Reshape (, 28 28)))
        (.summary)))


(defn build-autoencoder [encoder decoder]
      "Sequential((x) -> z, (z, y) -> x)"
      (def x (Input (, 28 28)))
      (def y (Input (, 10)))
      (setv z (encoder x)
            restructed-x (decoder [z y]))
      (doto
        (Model [x y] restructed-x)
        (.summary)))


(defn build-fake-autoencoder [encoder decoder]
      "Sequential((x) -> z, (z) -> x)"
      (def x (Input (, 28 28)))
      (setv z (encoder x)
            restructed-x (decoder z))
      (doto
        (Model x restructed-x)
        (.summary)))


(defn array->image [array]
      (Image.fromarray (.astype (* array 255.0) numpy.uint8)))


(defn array->ascii [array]
      (defn x->chr [x]
            (cond [(< x 0.2) " "]
                  [(< x 0.4) "."]
                  [(< x 0.6) ";"]
                  [True      "#"]))
      (.join "\n"
             (map
               (fn [line] (.join "" (map x->chr line)))
               array)))


(with-decorator
  (click.group)
  (defn cli [] 0))


(with-decorator
  (cli.command)
  (defn train []

        (defn minus_binary_crossentropy [y-true y-pred]
              (- (K.mean (K.binary_crossentropy y-pred y-true) :axis -1)))

        (def time (int (.timestamp (datetime.datetime.now))))

        (setv [[x-train y-train] [x-test y-test]] (load-data)
              encoder (build-encoder)
              decoder (build-decoder)
              fake-decoder (build-fake-decoder)
              autoencoder (build-autoencoder encoder decoder)
              fake-autoencoder-front (build-fake-autoencoder encoder fake-decoder))

        (def strong-opt (Adam :lr 0.003 :clipvalue 0.5)
             normal-opt (Adam :lr 0.001 :clipvalue 0.2)
             weak-opt   (Adam :lr 0.0005 :clipvalue 0.1))

        ; autoencoder
        (setv encoder.trainable True)
        (setv decoder.trainable True)
        (.compile autoencoder :loss 'binary_crossentropy :optimizer strong-opt)

        ; fake-autoencoder
        (setv encoder.trainable True)
        (setv fake-decoder.trainable False)
        (.compile fake-autoencoder-front :loss minus_binary_crossentropy :optimizer strong-opt)

        (setv encoder.trainable False)
        (setv fake-decoder.trainable True)
        (.compile fake-decoder :loss 'binary_crossentropy :optimizer normal-opt)

        ; Variational
        (setv encoder.trainable True)
        (.compile encoder :loss 'mse :optimizer weak-opt)

        (def batch-size 50)
        (def ten-vector (numpy.zeros (, batch-size 10)))  ; for style-transfer
        (for [i (range batch-size)]
             (setv (get ten-vector i (% i 10)) 1.0))
        (def zero-vector (numpy.zeros (, batch-size HIDDEN-DIM)))  ; for Variational

        ; training loop
        (for [epoch (range 10)
              index (range (// (len x-train) batch-size))]

             (def x-batch (get x-train (slice (* batch-size index) (* batch-size (inc index))))
                  y-batch (get y-train (slice (* batch-size index) (* batch-size (inc index))))
                  z-batch (.predict_on_batch encoder x-batch))

             ; training
             (for [_ (range 10)]
                  (.train_on_batch fake-decoder z-batch x-batch))

             (def loss-autoencoder (.train_on_batch autoencoder [x-batch y-batch] x-batch)
                  loss-fake-autoencoder (.train_on_batch fake-autoencoder-front [x-batch] x-batch)
                  ; loss-variational (.train_on_batch encoder x-batch zero-vector))
                  )

             ; log
             (setv loss {'loss-autoencoder (float loss-autoencoder)
                         'loss-fake-autoencoder (float loss-fake-autoencoder)
                         ; 'loss-variational (float loss-variational)
                         })


             (if (= (% index 100) 0)
               (do

                 (print [epoch index loss])
                 (with [f (open (.format "log/{}.jsonl" time) "a")]
                       (.write f (json.dumps loss))
                       (.write f "\n"))

                 (.save_weights encoder "snapshots/encoder.h5")
                 (.save_weights decoder "snapshots/decoder.h5")
                 (.save_weights fake-decoder "snapshots/fake-decoder.h5")

                 ; decoding
                 (def reconstructed-x (.predict_on_batch decoder [z-batch y-batch])
                      fake-reconstructed-x (.predict_on_batch fake-decoder z-batch)
                      generated-x (.predict_on_batch decoder [z-batch ten-vector]))

                 (print (array->ascii (get x-batch 0)))
                 (print (array->ascii (get reconstructed-x 0)))
                 (print (array->ascii (get generated-x 0)))

                 (for [i (range 10)]
                      (.save (array->image (get reconstructed-x i))
                             (.format "result/reconstructed.{}.{}.gif" epoch i))
                      (.save (array->image (get fake-reconstructed-x i))
                             (.format "result/fake-reconstructed.{}.{}.gif" epoch i))
                      (.save (array->image (get generated-x i))
                             (.format "result/generated.{}.{}.gif" epoch i)))))
        )))


(with-decorator
  (cli.command)
  (defn test []

        (setv [[x-train y-train] [x-test y-test]] (load-data)
              encoder (build-encoder)
              decoder (build-decoder))

        (.load_weights encoder "snapshots/encoder.h5")
        (.load_weights decoder "snapshots/decoder.h5")

        (def X (numpy.zeros (, 10 28 28)))

        (for [i (range 10)]
             (.save (array->image (get x-test i)) (.format "result/original.{}.gif" i))
             (setv (get X i) (get x-test i)))

        (def Z (.predict_on_batch encoder X))

        ; transfer the digit to `i`
        (for [i (range 10)]
             (def Y (numpy.zeros (, 10 10)))
             (for [j (range 10)] (setv (get Y j i) 1.0))
             (def re-X (.predict_on_batch decoder [Z Y]))
             (for [j (range 10)]
                  (.save (array->image (get re-X j)) (.format "result/trans.{}.{}.gif" j i))))))



(defmain [&rest args] (cli))
