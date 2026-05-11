import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io, os

U_inf = 30 # m/s
H = 1 # m
W1 = 0.1 # m
W2 = 0.3 # m
rho = 1.0 # kg/m^3
sobrerelaxacao = 1.85
tol = 0.01
max_domain_width = 9 * W2 # m
max_domain_height = 2 * H # m
dx=dy=0.01

# Gerando os vetores de coordenadas
x = np.arange(0, max_domain_width + dx, dx)
y = np.arange(0, max_domain_height + dy, dy)

# Criando a malha (Grid)
X, Y = np.meshgrid(x, y)

# Inicializando a matriz de Potencial Psi com zeros
# Usamos a transposta ou shape correto para bater com X e Y
psi = np.zeros_like(X)

def make_mask(X, Y, x_G, y_G, H, W1, W2, theta, show=False):

    theta_rad = np.radians(theta) # Ângulo de ataque (negativo para inclinar a asa)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    mask = np.zeros_like(X, dtype=bool)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            curr_x = X[i, j]
            curr_y = Y[i, j]
            
            # 1. Translada para a origem (centro da placa)
            xt = curr_x - x_G
            yt = curr_y - y_G
            
            # 2. Rotaciona o ponto para o referencial "reto" da placa
            x_rot = xt * cos_t - yt * sin_t
            y_rot = xt * sin_t + yt * cos_t
            
            # equação de formação do trapézio
            if -H/2 <= y_rot <= H/2:
                x_esq = (-W1) + ((-W2) - (-W1)) * (y_rot - (-H/2)) / H
                if x_esq <= x_rot <= 0: 
                    mask[i, j] = True
        if show:
            plt.figure(figsize=(8,4))
            plt.contourf(X, Y, mask, cmap='Greys')
            plt.title("Máscara da Placa (Preto = Sólido)")
            plt.colorbar()
            plt.show()
            plt.close()
    return mask


def make_psi(X, Y, U_inf, theta, mask, gif=False):
    
    # 1. Condição nas bordas do domínio + chute inicial
    psi = U_inf * Y  # Isso já preenche tudo seguindo a velocidade livre

    # 2. Condição na superfície da placa (Não penetração: psi = 0)
    psi[mask] = 0

    if gif:
        frames = [] # Lista para guardar as imagens
        # Configuração do gráfico para o GIF
        fig, ax = plt.subplots(figsize=(10, 5))
    # Parâmetros do Solver
    erro = 1.0
    iteracao = 0

    while erro > tol:
        psi_antigo = psi.copy()
        print(f"Iteração {iteracao} - Erro: {erro:.6f}")
        # Atualiza apenas o interior (do índice 1 até o penúltimo)
        # A fórmula abaixo calcula a média dos vizinhos para todos os pontos de uma vez
        for i in range(1, psi.shape[0] - 1):
            for j in range(1, psi.shape[1] - 1):
                #print(f"Atualizando ponto ({i}, {j}) - Máscara: {mask[i, j]}")
                if not mask[i, j]:
                    # Aqui, psi[i-1, j] já foi atualizado nesta iteração!
                    media = 0.25 * (psi[i+1, j] + psi[i-1, j] + psi[i, j+1] + psi[i, j-1])
                    psi[i, j] = (1 - sobrerelaxacao) * psi[i, j] + sobrerelaxacao * media
        
        # REAPLICA AS CONDIÇÕES FIXAS:
        # O contorno do domínio (bordas externas) já está fixo pois o slice [1:-1] não os toca
        # A placa deve continuar sendo zero (ela pode ter mudado no cálculo acima)
        psi[mask] = 0

        # Calcula o erro máximo (diferença absoluta)
        erro = np.max(np.abs(psi - psi_antigo))

        if gif:
            if iteracao % 10 == 0:
                ax.clear()
                # Plot das linhas de corrente atuais

                ax.set_facecolor('#D3D3D3')
                ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['#8B0000'], zorder=2) # asa
                ax.contour(X, Y, mask, levels=[0.5], colors=['black'], linewidths=1, zorder=3) # contorno da asa
                niveis = np.linspace(0, U_inf * max_domain_height, 50)
                ax.contour(X, Y, psi, levels=niveis, colors='blue', linewidths=0.7, zorder=1) # linhas de corrente

                # 4. Títulos e Eixos
                ax.set_title(f"Evolução do Escoamento $\psi$ (Ângulo {90+theta}° $U_\infty$ = {U_inf} m/s) \n Iteração: {iteracao} | Erro: {erro:.4f}")
                ax.set_xlabel("Distância X [m]")
                ax.set_ylabel("Distância Y [m]")
                ax.axis('equal')
                
                # Salva o gráfico atual na memória (Buffer)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                frames.append(Image.open(buf))
                
                print(f"Frame salvo - Iteração {iteracao}")

        iteracao += 1
        if iteracao % 100 == 0:
            print(f"Iteração {iteracao} - Erro: {erro:.6f}")
    print(f"Convergência atingida em {iteracao} iterações!")

    fig_final, ax_f = plt.subplots(figsize=(10, 5))

    ax_f.set_facecolor('#D3D3D3')
    ax_f.contourf(X, Y, mask, levels=[0.5, 1], colors=['#8B0000'], zorder=2) # asa
    ax_f.contour(X, Y, mask, levels=[0.5], colors=['black'], linewidths=1, zorder=3) # contorno da asa
    niveis = np.linspace(0, U_inf * max_domain_height, 50)
    ax_f.contour(X, Y, psi, levels=niveis, colors='blue', linewidths=0.7, zorder=1) # linhas de corrente

    ax_f.set_title(f"Linhas de Corrente - $\psi$ (Ângulo {90+theta}° $U_\infty$ = {U_inf} m/s)")
    ax_f.set_xlabel("Distância X [m]")
    ax_f.set_ylabel("Distância Y [m]")
    ax_f.axis('equal') # Mantém a proporção real 1:1

    diretorio_saida = "resultados2"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")
    # Salva a imagem final em alta resolução (PNG)
    caminho_imagem = f"{diretorio_saida}/resultado_final_{90+theta}deg.png"
    fig_final.savefig(caminho_imagem, dpi=600, bbox_inches='tight')
    print(f"Imagem final salva em: {caminho_imagem}")
    plt.close(fig_final)  # Fecha a figura usada para a imagem final
    
    if gif:
        # Verifica se a pasta existe; se não, cria automaticamente
        diretorio_saida = "resultados2"
        if not os.path.exists(diretorio_saida):
            os.makedirs(diretorio_saida)
            print(f"Pasta '{diretorio_saida}' criada com sucesso!")

        # 3. Salvando o arquivo GIF
        if frames:
            frames[0].save(f'{diretorio_saida}/asa_{90+theta}deg_escoamento_evoluindo.gif',
                    save_all=True, append_images=frames[1:], 
                    optimize=False, duration=100, loop=0)
        print("GIF gerado com sucesso!")
        frames.clear()
        plt.close(fig)        # Fecha a figura usada para o GIF
    return psi

def make_gradientes(psi, mask, dy=dy, dx=dx):
    pass
    # grad_y é a derivada em relação a y, grad_x em relação a x
    grad_y, grad_x = np.gradient(psi, dy, dx)

    u_x = grad_y      # Velocidade horizontal
    u_y = -grad_x     # Velocidade vertical
    V_abs = np.sqrt(u_x**2 + u_y**2) # Magnitude da velocidade
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define o fundo e a placa (mesmo estilo que você já usa)
    ax.set_facecolor('#D3D3D3')
    ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['#8B0000'], zorder=3)

    # Define o 'passo' das setas (ex: plotar a cada 5 ou 10 pontos)
    passo = 10

    # Criar o campo de vetores
    # X, Y: coordenadas; u_x, u_y: componentes; V_abs: define a cor das setas
    q = ax.quiver(X[::passo, ::passo], Y[::passo, ::passo], 
                u_x[::passo, ::passo], u_y[::passo, ::passo], 
                V_abs[::passo, ::passo], 
                cmap='jet', zorder=2)

    # Adiciona uma barra de cores para a velocidade
    plt.colorbar(q, label='Velocidade Absoluta [m/s]')

    ax.set_title("Campo de Vetores de Velocidade")
    ax.axis('equal')
    plt.show()

def main():
    theta = 0
    mask = make_mask(X, Y, max_domain_width / 2, max_domain_height / 2, H, W1, W2, theta=theta)
    psi = make_psi(X, Y, U_inf, theta, mask, gif=True)
    make_gradientes(psi, mask, dy, dx)

if __name__ == "__main__":
    main()