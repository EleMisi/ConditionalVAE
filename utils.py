import pickle
import tensorflow as tf



#######################
# Train Step Function #
#######################


def train_step(data, model, optimizer):


    with tf.GradientTape() as tape:

        model_output = model(data, is_train = True)

    trainable_variables = model.trainable_variables
    grads = tape.gradient(model_output['loss'], trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))

    total_loss = model_output['loss'].numpy().mean()
    recon_loss = model_output['reconstr_loss'].numpy().mean()
    latent_loss = model_output['latent_loss'].numpy().mean()

    return total_loss, recon_loss, latent_loss



##########################################
#   Utils to save and read pickle files  #
##########################################

def save_data(file_name, data):
    """
    Saves data on file_name.pickle.
    """
    with open((file_name+'.pickle'), 'wb') as openfile:
        print(type(data))
        pickle.dump(data, openfile)


def read_data(file_name):
    """
    Reads file_name.pickle and returns its content.
    """
    with (open((file_name+'.pickle'), "rb")) as openfile:
        while True:
            try:
                objects=pickle.load(openfile)
            except EOFError:
                break
    return objects
