
def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None, device_list='0'):
    """

    :param allow_growth: When necessary, reserve memory
    :param float per_process_gpu_memory_fraction: specify GPU memory usage as 0 to 1

    :return:
    """
    import tensorflow as tf
    
    # TensorFlow 2.x configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set visible GPUs
            tf.config.set_visible_devices(gpus[int(device_list)], 'GPU')
            
            # Configure GPU memory
            if per_process_gpu_memory_fraction is not None:
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=int(per_process_gpu_memory_fraction * gpu.memory_limit))])
            elif allow_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
