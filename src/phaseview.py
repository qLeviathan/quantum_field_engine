# --- PhaseView: Token Phase Animation Tool ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def phaseview_animation(phase_history, output_path="./phaseview_animation.mp4", fps=30):
    """
    Create an animation of phase evolution over time.

    Args:
        phase_history (list of list or array): Phase angles at each training step.
        output_path (str): Where to save the animation.
        fps (int): Frames per second.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, len(phase_history[0]))
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Phase (radians)")
    ax.set_title("PhaseView: Token Evolution Over Time")
    ax.grid(True)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        x = np.arange(len(phase_history[i]))
        y = phase_history[i]
        line.set_data(x, y)
        return (line,)

    ani = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(phase_history), interval=1000/fps, blit=True)
    
    # For MP4 files, try ffmpeg but have a fallback
    if output_path.endswith('.mp4'):
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps)
            plt.close()
            print(f"🎥 PhaseView animation saved to {output_path}")
        except (ValueError, KeyError) as e:
            # Handle ffmpeg not available
            gif_path = output_path.replace('.mp4', '.gif')
            print(f"⚠️ ffmpeg not available: {e}. Falling back to GIF format: {gif_path}")
            ani.save(gif_path, writer='pillow', fps=fps)
            plt.close()
            print(f"🎥 PhaseView animation saved to {gif_path}")
    
    # For GIF files, use pillow directly
    elif output_path.endswith('.gif'):
        ani.save(output_path, writer='pillow', fps=fps)
        plt.close()
        print(f"🎥 PhaseView animation saved to {output_path}")
    
    # For any other extension
    else:
        # Default to something safe
        gif_path = output_path.split('.')[0] + '.gif'
        ani.save(gif_path, writer='pillow', fps=fps)
        plt.close()
        print(f"🎥 PhaseView animation saved to {gif_path}")
