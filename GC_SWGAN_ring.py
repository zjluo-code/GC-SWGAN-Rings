######################
def generate_and_save_images(model,step, test_input, figure_size=(12,12), subplot=(10,10), save=False, is_flatten=False):
    '''
    Generate images and plot it.
    '''
    predictions = model.predict(test_input)
    if is_flatten:
        predictions = predictions.reshape(-1, IMG_WIDTH, IMG_HEIGHT, CHANNELS).astype('float32')
    fig = plt.figure(figsize=figure_size)
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, 'image_at_epoch_{:04d}.png'.format(step)))
    plt.show()

########################################
def plot_real_and_save_images(images,start_img=0, figure_size=(12,12), subplot=(2,2), save=False, is_flatten=False):
    '''
    show real images and plot it.
    '''
    real_img = images[start_img:start_img+subplot[0]*subplot[1]]
    if is_flatten:
        real_img = real_img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, CHANNELS).astype('float32')
    fig = plt.figure(figsize=figure_size)
    for i in range(real_img.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        plt.imshow(real_img[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, 'real_image_begin_no_{:04d}.png'.format(start_img+1)))
    plt.show()    

num_examples_to_generate = 9

# We will reuse this seed overtime
sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])


#Define training step

@tf.function
def CWGAN_GP_train_d_unsup_step(unsup_image, batch_size, step):
    '''
        One discriminator training step
    '''
    print("retrace")
    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_unsup_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator(noise, training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(unsup_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred  = discriminator_unsup(fake_image_mixed, training=True)

        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

        fake_pred = discriminator_unsup(fake_image, training=True)
        unsup_pred = discriminator_unsup(unsup_image, training=True)

        D_unsup_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(unsup_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_unsup_gradients = d_unsup_tape.gradient(D_unsup_loss,
                                              discriminator_unsup.trainable_variables)
    # Apply the gradients to the optimizer
    D_unsup_optimizer.apply_gradients(zip(D_unsup_gradients,
                                          discriminator_unsup.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('D_unsup_loss', tf.reduce_mean(D_unsup_loss), step=step)
    return D_unsup_loss


@tf.function
def CWGAN_GP_train_g_step(unsup_image, batch_size, step):
    '''
        One generator training step
    '''
    print("retrace")
    noise = tf.random.normal([batch_size, NOISE_DIM])
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        fake_image = generator(noise, training=True)
        fake_pred = discriminator_unsup(fake_image, training=True)
        G_loss  = -tf.reduce_mean(fake_pred)
        # Calculate the gradients for generator
    G_gradients = g_tape.gradient(G_loss,
                                      generator.trainable_variables)
    # Apply the gradients to the optimizer
    G_optimizer.apply_gradients(zip(G_gradients,
                                      generator.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('G_loss', G_loss, step=step)
    return G_loss

@tf.function
def CWGAN_GP_train_d_sup_step(sup_image,label_oh,batch_size, step):
    with tf.GradientTape(persistent=True) as d_sup_tape:
        label_pred = discriminator_sup(sup_image,training=True)
        D_sup_loss = tf.reduce_mean(tf.keras.losses.CategoricalCrossentropy(from_logits=False)(label_oh,label_pred))
    D_sup_gradients = d_sup_tape.gradient(D_sup_loss,
                                          discriminator_sup.trainable_variables)
    #tf.print("D_sup_loss:", D_sup_loss)
    D_sup_optimizer.apply_gradients(zip(D_sup_gradients,
                                        discriminator_sup.trainable_variables))
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('D_sup_loss', D_sup_loss, step=step)
    return D_sup_loss


####################################################################################

# Start training

current_learning_rate = LR
trace = True

for step in range(CURRENT_STEP+1, STEPs + 1):   
    start = time.time()
    print('Start of epoch %d' % (step,))
    # Using learning rate decay
    current_learning_rate = learning_rate_decay(current_learning_rate)
    print('current_learning_rate %f' % (current_learning_rate,))
    set_learning_rate(current_learning_rate)

    sup_image,label = dataset.batch_labeled(BATCH_SIZE)
    label_oh = to_categorical(label,num_classes = NCLASSES)

    unsup_image = dataset.batch_unlabeled(BATCH_SIZE)

    D_sup_loss = CWGAN_GP_train_d_sup_step(sup_image,label_oh,batch_size=tf.constant(BATCH_SIZE, dtype=tf.int64),step=tf.constant(step,dtype=tf.int64))

    D_unsup_loss = CWGAN_GP_train_d_unsup_step(unsup_image,batch_size=tf.constant(BATCH_SIZE, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))

    # Train generator

    G_loss = CWGAN_GP_train_g_step(unsup_image,batch_size= tf.constant(BATCH_SIZE, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))

    if step % 10 == 0:
        print ('.', end='')


    if step % SAVE_EVERY_N_STEP == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for step {} at {}'.format(step,
                                                            ckpt_save_path))

    print ('Time taken for step {} is {} sec\n'.format(step,
                                                       time.time()-start))

    print("D_sup_loss = ", D_sup_loss.numpy())
    print("D_unsup_loss = ", D_unsup_loss.numpy())
    print("G_loss = ", G_loss.numpy())
    lossfile.write("%10.5f %10.5f %10.5f\n" % (D_sup_loss,D_unsup_loss,G_loss))


step =STEPs #template !!!!!!!!!!!!

generate_and_save_images(generator,step, sample_noise, figure_size=(12,12), subplot=(3,3), save=False, is_flatten=False)

image_test,label_test = dataset.test_set()

print(image_test.shape)

test_pred = discriminator_sup.predict(image_test)

#for row in range(len(test_pred)):
y_pred = np.argmax(test_pred,axis=1)
print(test_pred.shape)
y_true = label_test


# caculate conf_matrix

conf_matrix = confusion_matrix(y_true, y_pred)
cm_normalized = conf_matrix.astype('float')/conf_matrix.sum(axis=1)[:,np.newaxis]
# print conf_matrix
print("Confusion matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion matrix')
plt.show()
# Calculate precision, recall and F1 Score
report = classification_report(y_true, y_pred, output_dict=True)
#print(report.items())


print("Macro Average:")  
print(f"  Precision: {report['macro avg']['precision']:.4f}")
print(f"  Recall: {report['macro avg']['recall']:.4f}")
print(f"  F1 score: {report['macro avg']['f1-score']:.4f}")
print()

print("Weighted Average:")  
print(f"  Precision: {report['weighted avg']['precision']:.4f}")
print(f"  Recall: {report['weighted avg']['recall']:.4f}")
print(f"  F1 score: {report['weighted avg']['f1-score']:.4f}")

for label, metrics in report.items():
    #if isinstance(label, int):
        print(f"class {label}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1-score']:.4f}")
        print()
