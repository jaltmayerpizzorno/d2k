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


detection* d2k_get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num) {
    #if PJREDDIE
        return get_network_boxes(net, w, h, thresh, hier, map, relative, num);
    #else
        static const int letter = 1;    // whether image was "letterboxed" to resize maintaining aspect ratio
        return get_network_boxes(net, w, h, thresh, hier, map, relative, num, letter);
    #endif
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

int has_input(layer* l, int n) {
    for (int i=0; i<l->n; i++) {
        if (l->input_layers[i] == n) {
            return 1;
        }
    }

    return 0;
}

typedef struct {
    int h, w, c;
    float* output;
} layer_output;

typedef struct {
    int count;
    layer_output* output;

} net_outputs;


net_outputs get_net_outputs(network* net) {
    int indices[net->n];
    int count = 0;

    int i = net->n-1;

    for(; i >= 0; --i) {
        // this is as in darknet's get_network_output_layer
        if (net->layers[i].type != COST) {
            indices[count++] = i;
            break;
        }
    }

    for(; i > 0; --i) {
        if (net->layers[i].type == ROUTE) {
            // if previous layer isn't an input to the route, it's a network output (leaf on tree)
            if (!has_input(&net->layers[i], i-1)) {
                indices[count++] = i-1;
            }
        }
    }

    net_outputs outputs;
    outputs.count = count;
    outputs.output = calloc(count, sizeof(layer_output));

    for (i=0; i<count; ++i) {
        layer l = net->layers[indices[count-i-1]];

        outputs.output[i].h = l.out_h;
        outputs.output[i].w = l.out_w;
        outputs.output[i].c = l.out_c;
        outputs.output[i].output = l.output;
    }

    return outputs;
}

void free_net_outputs(net_outputs netout) {
    free(netout.output);
}
