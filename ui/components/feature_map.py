from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import cv2

from ui.utils.session_state_service import SessionStateService

class FeatureMap:
    def render(self):
        conv1_out = SessionStateService.get('conv1_out')
        relu_out_1 = SessionStateService.get('relu_out_1')
        conv2_out = SessionStateService.get('conv2_out')
        relu_out_2 = SessionStateService.get('relu_out_2')
        conv3_out = SessionStateService.get('conv3_out')
        relu_out_3 = SessionStateService.get('relu_out_3')
        model = SessionStateService.get('model')
        original_image = SessionStateService.get('image')
        
        
        st.header("🎨 Feature Map Visualization")
        st.markdown("""
        See how the neural network processes images through its layers.
        Each feature map shows what patterns and features the network detects.
        
        Understanding the colors:
        - Bright colors (yellow/white) = Strong feature detection
        - Dark colors (purple/black) = Weak or no feature detection
        - Color intensity indicates detection strength
        """)

        if model is not None:
            with st.expander("Learned Filters", expanded=False):
                self.visualize_filters(model, "conv1")

        if original_image is not None:
            with st.expander("Activation Heatmaps", expanded=False):
                self.visualize_heatmaps(conv1_out, original_image, "First Convolutional Layer")
                if conv2_out is not None:
                    self.visualize_heatmaps(conv2_out, original_image, "Second Convolutional Layer")
                if conv3_out is not None:
                    self.visualize_heatmaps(conv3_out, original_image, "Third Convolutional Layer")

        # 1. Activations after convolutional layer
        conv_features_1 = conv1_out[0].detach().numpy()
        n_filters = conv_features_1.shape[0]

        st.expander("Layer 1: Initial Feature Detection", expanded=False)

        with st.expander("Layer 1: Initial Feature Detection", expanded=False):
            st.subheader("Convolutional Layer Output")
            st.markdown("""
            *First layer feature detection*
            
            The network starts by identifying basic visual elements:
            - Edges and lines
            - Simple textures and patterns
            - Basic contrasts in the image
            """)
            
            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                # Add selectbox for filter selection
                selected_filter = st.selectbox(
                    "Select filter to visualize",
                    range(n_filters),
                    key="layer1_filter_select",
                    format_func=lambda x: f"Filter {x+1}"
                )
                
                # Show selected filter with smaller size
                fig = plt.figure(figsize=(2, 2))  # Reduced from (4, 4)
                plt.imshow(conv_features_1[selected_filter], cmap='magma')
                plt.title(f'Filter {selected_filter + 1}', pad=5, fontsize=8)  # Smaller title
                plt.axis('off')
                st.pyplot(fig, use_container_width=False)  # Don't use full container width
                st.caption(f"Filter {selected_filter + 1} detection pattern")
            
            # 2. Activations after LeakyReLU
            relu_features = relu_out_1[0].detach().numpy()
            
            with col2:
                st.subheader("LeakyReLU Layer Output")
                st.markdown("""
                *Feature enhancement with LeakyReLU*
                
                The LeakyReLU layer:
                - Preserves important detected features
                - Allows small negative values to pass through
                - Helps prevent "dying neurons" problem
                - Creates smoother gradient flow
                """)
                
                # Show the same filter after LeakyReLU with smaller size
                fig = plt.figure(figsize=(2, 2))  # Reduced from (4, 4)
                plt.imshow(relu_features[selected_filter], cmap='magma')
                plt.title(f'Filter {selected_filter + 1}', pad=5, fontsize=8)  # Smaller title
                plt.axis('off')
                st.pyplot(fig, use_container_width=False)  # Don't use full container width
                st.caption(f"Enhanced Filter {selected_filter + 1}")

        with st.expander("Layer 2: Intermediate Feature Detection", expanded=False):
            if conv2_out is not None:
                conv_features_2 = conv2_out[0].detach().numpy()
                n_filters_2 = conv_features_2.shape[0]
                
                # Create two columns for side-by-side display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Convolutional Layer Output")
                    st.markdown("""
                    *Second layer pattern recognition*
                    
                    The network now combines basic features to detect:
                    - More complex shapes
                    - Pattern combinations
                    - Spatial relationships
                    """)    

                    # Add selectbox for filter selection
                    selected_filter_2 = st.selectbox(
                        "Select filter to visualize",
                        range(n_filters_2),
                        key="layer2_filter_select",
                        format_func=lambda x: f"Filter {x+1}"
                    )
                    
                    # Show selected filter with smaller size
                    fig = plt.figure(figsize=(2, 2))  # Reduced from (4, 4)
                    plt.imshow(conv_features_2[selected_filter_2], cmap='magma')
                    plt.title(f'Filter {selected_filter_2 + 1}', pad=5, fontsize=8)  # Smaller title
                    plt.axis('off')
                    st.pyplot(fig, use_container_width=False)  # Don't use full container width
                    st.caption(f"Filter {selected_filter_2 + 1} detection pattern")
                
                with col2:
                    # Activations after LeakyReLU
                    relu_features_2 = relu_out_2[0].detach().numpy()
                    
                    st.subheader("LeakyReLU Layer Output")
                    st.markdown("""
                    *Enhanced intermediate features with LeakyReLU*
                    
                    Further refinement of detected patterns while maintaining:
                    - Small negative gradients
                    - Better feature propagation
                    - Improved learning stability
                    """)    

                    # Show the same filter after LeakyReLU with smaller size
                    fig = plt.figure(figsize=(2, 2))  # Reduced from (4, 4)
                    plt.imshow(relu_features_2[selected_filter_2], cmap='magma')
                    plt.title(f'Filter {selected_filter_2 + 1}', pad=5, fontsize=8)  # Smaller title
                    plt.axis('off')
                    st.pyplot(fig, use_container_width=False)  # Don't use full container width
                    st.caption(f"Enhanced Filter {selected_filter_2 + 1}")

        with st.expander("Layer 3: Advanced Feature Detection", expanded=False):
            if conv3_out is not None:
                conv_features_3 = conv3_out[0].detach().numpy()
                n_filters_3 = conv_features_3.shape[0]
                
                # Create two columns for side-by-side display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Convolutional Layer Output")
                    st.markdown("""
                    *Third layer high-level feature detection*
                    
                    The network identifies:
                    - Complex shape combinations
                    - Abstract patterns
                    - High-level visual features
                    """)        

                    # Add selectbox for filter selection
                    selected_filter_3 = st.selectbox(
                        "Select filter to visualize",
                        range(n_filters_3),
                        key="layer3_filter_select",
                        format_func=lambda x: f"Filter {x+1}"
                    )
                    
                    # Show selected filter with smaller size
                    fig = plt.figure(figsize=(2, 2))  # Reduced from (4, 4)
                    plt.imshow(conv_features_3[selected_filter_3], cmap='magma')
                    plt.title(f'Filter {selected_filter_3 + 1}', pad=5, fontsize=8)  # Smaller title
                    plt.axis('off')
                    st.pyplot(fig, use_container_width=False)  # Don't use full container width
                    st.caption(f"Filter {selected_filter_3 + 1} detection pattern")

                with col2:
                    # Activations after LeakyReLU
                    relu_features_3 = relu_out_3[0].detach().numpy()
                    
                    st.subheader("LeakyReLU Layer Output")
                    st.markdown("""
                    *Enhanced high-level features with LeakyReLU*
                    
                    Final refinement of complex patterns while maintaining:
                    - Small negative gradients
                    - Better feature propagation
                    - Improved learning stability
                    - Enhanced high-level feature detection
                    """)    

                    # Show the same filter after LeakyReLU with smaller size
                    fig = plt.figure(figsize=(2, 2))  # Reduced from (4, 4)
                    plt.imshow(relu_features_3[selected_filter_3], cmap='magma')
                    plt.title(f'Filter {selected_filter_3 + 1}', pad=5, fontsize=8)  # Smaller title
                    plt.axis('off')
                    st.pyplot(fig, use_container_width=False)  # Don't use full container width
                    st.caption(f"Enhanced Filter {selected_filter_3 + 1}")

    def visualize_filters(self, model, layer_name):
        """Visualize the learned filters in a convolutional layer"""
        st.subheader(f"{layer_name} Learned Filters")
        
        # Get the weights from the model
        if layer_name == "conv1":
            weights = model.conv1.weight.data.cpu().numpy()
        elif layer_name == "conv2":
            weights = model.conv2.weight.data.cpu().numpy()
        elif layer_name == "conv3":
            weights = model.conv3.weight.data.cpu().numpy()
            
        # Normalize weights for visualization
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        # Get number of filters
        n_filters = weights.shape[0]
        filters_per_row = 6
        
        st.markdown("""
        *Learned filter visualization*
        
        These are the actual patterns that the network has learned to detect.
        Each filter represents a specific feature detector.
        """)
        
        for row in range(0, n_filters, filters_per_row):
            row_filters = min(filters_per_row, n_filters - row)
            cols = st.columns(row_filters)
            for i in range(row_filters):
                with cols[i]:
                    # For first layer, show RGB channels
                    if weights.shape[1] == 3:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        # Combine RGB channels for visualization
                        filter_img = np.mean(weights[row + i], axis=0)
                        ax.imshow(filter_img, cmap='gray')
                        ax.set_title(f'Filter {row + i + 1}')
                        ax.axis('off')
                        st.pyplot(fig)
                    else:
                        # For deeper layers, show each channel
                        n_channels = weights.shape[1]
                        fig, axes = plt.subplots(1, n_channels, figsize=(4*n_channels, 4))
                        if n_channels == 1:
                            axes = [axes]  # Make axes iterable for single channel
                        for ch in range(n_channels):
                            axes[ch].imshow(weights[row + i, ch], cmap='gray')
                            axes[ch].axis('off')
                        st.pyplot(fig)
                    st.caption(f"Filter {row + i + 1} learned pattern")

    def create_heatmap(self, activations, original_image, layer_name, filter_idx=None):
        """Create a heatmap visualization"""
        # Convert to numpy if needed
        if hasattr(activations, 'detach'):
            activations = activations.detach().numpy()
        if hasattr(original_image, 'detach'):
            original_image = original_image.detach().numpy()
        elif hasattr(original_image, 'numpy'):
            original_image = original_image.numpy()
        elif hasattr(original_image, 'shape') is False:
            original_image = np.array(original_image)
            
        # If specific filter is requested
        if filter_idx is not None:
            heatmap = activations[0, filter_idx]
        else:
            # Combine all filters
            heatmap = np.mean(activations[0], axis=0)
            
        # Get image dimensions
        if len(original_image.shape) == 4:  # Batch, Channels, Height, Width
            height, width = original_image.shape[2], original_image.shape[3]
        elif len(original_image.shape) == 3:  # Channels, Height, Width
            height, width = original_image.shape[1], original_image.shape[2]
        else:  # Height, Width
            height, width = original_image.shape[0], original_image.shape[1]
            
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (width, height))
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(original_image.shape) == 4:
            img = original_image[0].transpose(1, 2, 0)
        elif len(original_image.shape) == 3:
            img = original_image.transpose(1, 2, 0)
        else:
            img = original_image
            
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Heatmap
        im = ax2.imshow(heatmap, cmap='jet')
        ax2.set_title('Activation Heatmap')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Overlay
        overlay = img.copy()
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        ax3.imshow(overlay, cmap='gray')
        ax3.imshow(heatmap, cmap='jet', alpha=0.5)
        ax3.set_title('Overlay')
        ax3.axis('off')
        
        plt.suptitle(f'{layer_name} Heatmap' + (f' - Filter {filter_idx}' if filter_idx is not None else ''))
        return fig

    def visualize_heatmaps(self, activations, original_image, layer_name):
        """Visualize heatmaps for a layer"""
        st.subheader(f"{layer_name} Activation Heatmaps")
        
        # Combined heatmap
        st.markdown("### Combined Heatmap")
        st.markdown("""
        This shows the overall activation pattern across all filters.
        Brighter areas indicate stronger activations.
        """)
        fig = self.create_heatmap(activations, original_image, layer_name)
        st.pyplot(fig)
        
        # Individual filter heatmaps
        st.markdown("### Individual Filter Heatmaps")
        st.markdown("""
        These show activation patterns for individual filters.
        Select a filter to see its specific pattern.
        """)
        
        n_filters = activations.shape[1]
        
        # Convert layer name to snake_case for the key
        layer_key = layer_name.lower().replace(" ", "_")
        selectbox_key = f"filter_select_{layer_key}"
        
        # Use the session state value in the selectbox
        selected_filter = st.selectbox(
            f"Select {layer_name} filter to visualize",
            range(n_filters),
            key=selectbox_key,
            format_func=lambda x: f"Filter {x+1}"
        )
        
        fig = self.create_heatmap(activations, original_image, layer_name, selected_filter)
        st.pyplot(fig)

    