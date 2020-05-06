#include "darknet/include/darknet.h"

// --- pjreddie / AlexeyAB compatibility layer

// XXX is there a better way to detect which version of darknet this is?
#if defined(LIB_API)
    # define PJREDDIE 0
#else
    # define PJREDDIE 1
#endif


int d2k_is_pjreddie() {
    return PJREDDIE;
}


int d2k_network_inputs(network* net) {
    #if PJREDDIE
        return net->inputs;
    #else
        return net->inputs;
    #endif
}


float* d2k_network_predict(network *net, float *input) {
    #if PJREDDIE
        return network_predict(net, input);
    #else
        return network_predict(*net, input);
    #endif
}


float* d2k_network_predict_image(network *net, image im) {
    #if PJREDDIE
        return network_predict_image(net, im);
    #else
        return network_predict_image_letterbox(net, im);
    #endif
}


detection* d2k_get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int do_nms) {
    #if PJREDDIE
        detection* boxes = get_network_boxes(net, w, h, thresh, hier, map, relative, num);
        if (do_nms && *num > 0) {
            do_nms_sort(boxes, *num, boxes[0].classes, thresh);
        }
    #else
        static const int letter = 1;    // whether image was "letterboxed" to resize maintaining aspect ratio
        detection* boxes = get_network_boxes(net, w, h, thresh, hier, map, relative, num, letter);

        if (do_nms && *num > 0) {
            layer* yolo = &net->layers[net->n-1];
            assert(yolo->type == YOLO);

            if (yolo->nms_kind == DEFAULT_NMS) {
                do_nms_sort(boxes, *num, yolo->classes, thresh);
            }
            else {
                diounms_sort(boxes, *num, yolo->classes, thresh, yolo->nms_kind, yolo->beta_nms);
            }
        }
    #endif

    return boxes;
}


void d2k_free_network(network* net) {
    #if PJREDDIE
        free_network(net);
    #else
        free_network(*net);
        free(net);
    #endif
}

// ---

typedef struct {
    int h, w, c;
    float* output;
} layer_output;

typedef struct {
    int count;
    layer_output* output;

} net_outputs;


/**
 * Collects the output of all all layers whose output isn't used by another layer
 */
net_outputs get_net_outputs(network* net) {
    int isOutput[net->n];
    memset(&isOutput, 0, sizeof(isOutput));

    // last layer isn't consumed by anyone, so it's an output
    isOutput[net->n-1] = 1;

    for (int i=1; i<net->n; i++) {  // start at 1 because a layer 0 [route] isn't valid
        if (net->layers[i].type == ROUTE) {
            layer* l = &net->layers[i];

            isOutput[i-1] = 1;  // unless it's an input below

            for (int j=0; j<l->n; j++) {
                isOutput[l->input_layers[j]] = 0;
            }
        }
    }

    int count = 0;

    for (int i=0; i<net->n; i++) {
        if (isOutput[i]) {
            ++count;
        }
    }

    net_outputs outputs;
    outputs.count = 0;
    outputs.output = calloc(count, sizeof(layer_output));

    for (int i=0; i<net->n; i++) {
        if (isOutput[i]) {
            layer* l = &net->layers[i];

            outputs.output[outputs.count].h = l->out_h;
            outputs.output[outputs.count].w = l->out_w;
            outputs.output[outputs.count].c = l->out_c;
            outputs.output[outputs.count].output = l->output;

            ++outputs.count;
        }
    }

    return outputs;
}

void free_net_outputs(net_outputs netout) {
    free(netout.output);
}
