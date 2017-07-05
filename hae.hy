(import
  [click]
  [json]
  [datetime]
  [numpy]
  [PIL [Image]]
  [keras.backend [tensorflow_backend :as K]]
  [keras.layers [Dense Flatten Reshape Input]]
  [keras.layers.convolutional [Conv2D UpSampling2D]]
  [keras.layers.merge [concatenate]]
  [keras.optimizers [Adam]]
  [keras.models [Sequential Model]]
  [keras.datasets [mnist]]
  [keras.utils [np_utils]])

(def HIDDEN-DIM 32)


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
       (.add (Conv2D 8 (, 5 5) :strides (, 2 2) :activation 'relu))
       (.add (Conv2D 16 (, 5 5) :strides (, 2 2) :activation 'relu))
       (.add (Flatten))
       (.add (Dense HIDDEN-DIM))
       (.summary)))


(defn build-decoder []
      "{hidden-dim 10-dim} => 28x28"
      (def z (Input (, HIDDEN-DIM)))
      (def y (Input (, 10)))
      (def reconstructed-x
           (-> [z ((Dense 10 :activation 'relu) y)]
               concatenate
               ((Dense (* 14 14) :activation 'relu))
               ((Reshape (, 14 14 1)))
               ((UpSampling2D :size (, 2 2)))
               ((Conv2D 1 5 :padding 'same))
               ((Reshape (, 28 28)))))
      (doto
        (Model [z y] reconstructed-x)
        (.summary)))


(defn build-fake-decoder []
      "hidden-dim => 28x28"
      (doto
        (Sequential :name 'fake-decoder)
        (.add (Dense (* 14 14) :activation 'relu :input_shape (, HIDDEN-DIM)))
        (.add (Reshape (, 14 14 1)))
        (.add (UpSampling2D :size (, 2 2)))
        (.add (Conv2D 1 5 :padding 'same))
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
              autoencoder (build-autoencoder encoder decoder)
              fake-decoder (build-fake-decoder)
              fake-autoencoder (build-fake-autoencoder encoder fake-decoder))

        (def strong-opt (Adam :lr 0.002 :clipvalue 1.0)
             normal-opt (Adam :clipvalue 0.5)
             weak-opt (Adam :lr 0.0005 :clipvalue 0.1))

        ; autoencoder
        (setv encoder.trainable True)
        (setv decoder.trainable True)
        (.compile autoencoder :loss 'binary_crossentropy :optimizer strong-opt)

        ; fake-autoencoder
        (setv fake-decoder.trainable True)
        (.compile fake-decoder :loss 'binary_crossentropy :optimizer normal-opt)

        ; independence
        (setv encoder.trainable True)
        (setv fake-decoder.trainable False)
        (.compile fake-autoencoder :loss minus_binary_crossentropy :optimizer normal-opt)

        ; Variational
        (setv encoder.trainable True)
        (.compile encoder :loss 'mse :optimizer weak-opt)

        (def batch-size 50)
        (def ten-vector (numpy.zeros (, batch-size 10)))  ; for style-transfer
        (for [i (range batch-size)]
             (setv (get ten-vector i (% i 10)) 1.0))
        (def zero-vector (numpy.zeros (, batch-size HIDDEN-DIM)))  ; for Variational

        ; training loop
        (for [epoch (range 100)
              index (range (// (len x-train) batch-size))]

             (def x-batch (get x-train (slice (* batch-size index) (* batch-size (inc index))))
                  y-batch (get y-train (slice (* batch-size index) (* batch-size (inc index))))
                  z-batch (.predict_on_batch encoder x-batch))

             ; training
             (def loss-autoencoder (.train_on_batch autoencoder [x-batch y-batch] x-batch)
                  loss-fake-decoder (.train_on_batch fake-decoder z-batch x-batch)
                  loss-fake-autoencoder (.train_on_batch fake-autoencoder [x-batch] x-batch)
                  loss-variational (.train_on_batch encoder x-batch zero-vector))

             ; log
             (setv loss {'loss-autoencoder (float loss-autoencoder)
                         'loss-fake-decoder (float loss-fake-decoder)
                         'loss-fake-autoencoder (float loss-fake-autoencoder)
                         'loss-variational (float loss-variational) })

             (print [epoch index loss])

             (with [f (open (.format "log/{}.jsonl" time) "a")]
                   (.write f (json.dumps loss))
                   (.write f "\n"))

             ; testing
             (if (= (% index 100) 0)
               (do
                 (print "snapshot saving...")
                 (.save_weights encoder "snapshots/encoder.h5")
                 (.save_weights decoder "snapshots/decoder.h5")
                 (.save_weights fake-decoder "snapshots/fake-decoder.h5")

                 ; decoding
                 (def reconstructed-x (.predict_on_batch decoder [z-batch y-batch])
                      fake-reconstructed-x (.predict_on_batch fake-decoder z-batch)
                      generated-x (.predict_on_batch decoder [z-batch ten-vector]))
                 (for [i (range 10)]
                      (.save (array->image (get reconstructed-x i))
                             (.format "result/reconstructed.{}.{}.gif" epoch i))
                      (.save (array->image (get fake-reconstructed-x i))
                             (.format "result/fake-reconstructed.{}.{}.gif" epoch i))
                      (.save (array->image (get generated-x i))
                             (.format "result/generated.{}.{}.gif" epoch i)))))
        )))

(defmain [&rest args] (cli))
