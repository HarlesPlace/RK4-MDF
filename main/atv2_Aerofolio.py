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
            frames[0].save(f'{diretorio_saida}/asa_vel{U_inf}_{90+theta}deg_escoamento_evoluindo.gif',
                    save_all=True, append_images=frames[1:], 
                    optimize=False, duration=100, loop=0)
        print("GIF gerado com sucesso!")
        frames.clear()
        plt.close(fig)        # Fecha a figura usada para o GIF
    return psi

def make_gradientes(psi, mask, U_inf, theta, dy=dy, dx=dx):
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
    distancia_entre_setas = 0.1 

    # O passo (inteiro) será essa distância dividida pelo tamanho do seu pixel
    # Usamos int() e max(1, ...) para garantir que o passo seja pelo menos 1
    passo = max(1, int(distancia_entre_setas / dx))

    # Criar o campo de vetores
    # X, Y: coordenadas; u_x, u_y: componentes; V_abs: define a cor das setas
    q = ax.quiver(X[::passo, ::passo], Y[::passo, ::passo], 
                u_x[::passo, ::passo], u_y[::passo, ::passo], 
                V_abs[::passo, ::passo], 
                cmap='jet', zorder=2)

    # Adiciona uma barra de cores para a velocidade
    plt.colorbar(q, label='Velocidade Absoluta [m/s]')

    ax.set_title(f"Campo de Vetores de Velocidade \n (Ângulo {90+theta}° $U_\infty$ = {U_inf} m/s)")
    ax.axis('equal')
    ax.set_xlabel("Distância X [m]")
    ax.set_ylabel("Distância Y [m]")
    ax.axis('equal') # Mantém a proporção real 1:1

    diretorio_saida = "resultados2"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada com sucesso!")
    # Salva a imagem final em alta resolução (PNG)
    caminho_imagem = f"{diretorio_saida}/gradientes-vel{U_inf}_{90+theta}deg.png"
    fig.savefig(caminho_imagem, dpi=600, bbox_inches='tight')
    print(f"Imagem final salva em: {caminho_imagem}")
    plt.close(fig)  # Fecha a figura usada para a imagem final

    return u_x, u_y, V_abs

def calc_pressure(u_x, u_y, U_inf, mask, theta):
    # Constantes
    rho = 1.0  # Conforme você passou anteriormente
    P_max_teorico = 0.5 * rho * U_inf**2
    
    # 1. Cálculo da Pressão
    V_quadrado = u_x**2 + u_y**2
    pressao = 0.5 * rho * (U_inf**2 - V_quadrado)

    # 2. Configuração do Gráfico (Interface orientada a objetos para melhor controle)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Mascarar a pressão dentro da asa
    pressao_plot = np.ma.masked_where(mask, pressao)
    
    # Níveis de pressão (focando na física e ignorando as quinas brancas)
    niveis = np.linspace(-5000, 1.2 * P_max_teorico, 100)
    
    # Plotar o mapa de pressões com 'extend' para pintar o que foge da escala
    cp = ax.contourf(X, Y, pressao_plot, levels=niveis, cmap='jet', extend='both')
    fig.colorbar(cp, label='Pressão Manométrica [Pa]')

    # Desenhar a asa por cima (zorder alto para acabamento)
    ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['black'], zorder=3)

    ax.set_title(f"Distribuição de Pressão - Ângulo {90+theta}° $U_\infty$ = {U_inf} m/s)")
    ax.set_xlabel("Distância X [m]")
    ax.set_ylabel("Distância Y [m]")
    ax.axis('equal')

    # 3. Lógica de Salvamento
    diretorio_saida = "resultados2"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Pasta '{diretorio_saida}' criada!")

    nome_arquivo = f"{diretorio_saida}/pressao_asa-vel{U_inf}_{90+theta}deg.png"
    # dpi=600 garante que as curvas fiquem nítidas no PDF do relatório
    fig.savefig(nome_arquivo, dpi=600, bbox_inches='tight')
    print(f"Figura de pressão salva em: {nome_arquivo}")

    
    # 4. Limpeza de memória
    plt.close(fig)

    return pressao, np.min(pressao), np.max(pressao)

def calc_forces(pressao, mask, dx, dy, rho, U_inf, theta):
    """
    Calcula Sustentação (L) e Arrasto (D) integrando a pressão na superfície.
    """
    # 1. Encontrar a 'fronteira' da asa
    # Usamos um truque de deslocamento de matriz para achar quem é vizinho do sólido
    is_fluid = ~mask
    
    # Vizinhos imediatos (Cima, Baixo, Esquerda, Direita)
    # Se o ponto é fluido mas tem um vizinho sólido, ele está na superfície
    borda_inferior = is_fluid & np.roll(mask,  1, axis=0) # Fluido logo abaixo do sólido
    borda_superior = is_fluid & np.roll(mask, -1, axis=0) # Fluido logo acima do sólido
    borda_esquerda = is_fluid & np.roll(mask,  1, axis=1) # Fluido à esquerda do sólido
    borda_direita  = is_fluid & np.roll(mask, -1, axis=1) # Fluido à direita do sólido

    # garantir que as bordas não se sobreponham
    borda_superior = borda_superior & ~borda_esquerda & ~borda_direita
    borda_inferior = borda_inferior & ~borda_esquerda & ~borda_direita
    # 2. Integrar as pressões (Soma de P * area)
    # Sustentação (L): Pressão de baixo empurra pra cima (+), de cima empurra pra baixo (-)
    # Multiplicamos por dx pois é a largura de cada 'pixel' da face
    L = (np.sum(pressao[borda_superior]) - np.sum(pressao[borda_inferior])) * dx
    
    # Arrasto (D): Pressão da frente empurra pra trás (+), de trás empurra pra frente (-)
    # Multiplicamos por dy pois é a altura de cada 'pixel' da face
    D = (np.sum(pressao[borda_esquerda]) - np.sum(pressao[borda_direita])) * dy

    # 3. Adimensionalizar (Coeficientes)
    # Corda média (c) pode ser aproximada pela largura média do trapézio ou W2
    c = H # Corda da asa (m)
    CL = L / (0.5 * rho * (U_inf**2) * c)
    CD = D / (0.5 * rho * (U_inf**2) * c)
    # 4. Configuração do Gráfico de Forças
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Desenha a asa (preto) e o fundo (cinza claro)
    ax.set_facecolor('#F0F0F0')
    ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['black'], zorder=1)
    
    # Coordenadas do Centro G
    x_G = (9 * W2) / 2
    y_G = (2 * H) / 2
    
    # Plotagem dos Vetores (Escalonados para visibilidade)
    # Ajuste o fator_escala se as setas ficarem muito pequenas ou grandes
    # 1. Definir o tamanho máximo visual da seta (ex: 25% da altura do gráfico)
    tamanho_max_seta = 0.25 * (2 * H) 
    
    # 2. Encontrar a maior força para servir de referência de escala
    maior_forca = max(abs(L), abs(D))
    
    # 3. Calcular o fator de escala dinâmico
    # O scale do quiver funciona assim: (Valor da Força) / (Escala) = Tamanho no gráfico
    # Logo, Escala = Maior Força / Tamanho Máximo Desejado
    escala_dinamica = maior_forca / tamanho_max_seta
    
    # Vetor Sustentação (Verde)
    ax.quiver(x_G, y_G, 0, L, color='green', angles='xy', scale_units='xy', 
              scale=escala_dinamica, width=0.015, label='Sustentação (L)', zorder=5)
    
    # Vetor Arrasto (Vermelho)
    ax.quiver(x_G, y_G, D, 0, color='red', angles='xy', scale_units='xy', 
              scale=escala_dinamica, width=0.015, label='Arrasto (D)', zorder=5)

    # 5. Caixa de Texto com os Coeficientes (Estilo 'Legend')
    texto_stats = (f'L: {L:.2f} N\n'
                   f'D: {D:.2f} N\n'
                   f'CL: {CL:.4f}\n'
                   f'CD: {CD:.4f}')
    
    # Posiciona a caixa no canto superior esquerdo
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, texto_stats, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, family='monospace')

    # Detalhes de formatação
    ax.set_title(f"Diagrama de Forças Aerodinâmicas - $\\theta = {90+theta}^\\circ$ $U_\infty$ = {U_inf} m/s)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')

    # 6. Lógica de Salvamento
    diretorio_saida = "resultados2"
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
    
    nome_figura = f"{diretorio_saida}/forcas_asa-vel{U_inf}_{90+theta}deg.png"
    fig.savefig(nome_figura, dpi=600, bbox_inches='tight')
    print(f"Gráfico de forças salvo: {nome_figura}")
    
    plt.close(fig)

    return L, D, CL, CD

# def calc_forces2(pressao, mask, dx, dy, rho, U_inf, theta):
#     surface = np.zeros_like(mask, dtype=bool)

#     for i in range(1, mask.shape[0]-1):
#         for j in range(1, mask.shape[1]-1):

#             if mask[i,j]:

#                 vizinhos = [
#                     mask[i+1,j],
#                     mask[i-1,j],
#                     mask[i,j+1],
#                     mask[i,j-1]
#                 ]

#                 if not all(vizinhos):
#                     surface[i,j] = True
#     ny, nx = np.gradient(mask.astype(float), dy, dx)
#     norma = np.sqrt(nx**2 + ny**2) + 1e-12

#     nx /= norma
#     ny /= norma
#     Fx = 0.0
#     Fy = 0.0

#     for i,j in np.argwhere(surface):

#         p = pressao[i,j]

#         dS = np.sqrt(dx**2 + dy**2)

#         Fx += -p * nx[i,j] * dS
#         Fy += -p * ny[i,j] * dS
    
#     D = Fx
#     L = Fy  
#     c = H # Corda da asa (m)
#     CL = L / (0.5 * rho * (U_inf**2) * c)
#     CD = D / (0.5 * rho * (U_inf**2) * c)
#     # 4. Configuração do Gráfico de Forças
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # Desenha a asa (preto) e o fundo (cinza claro)
#     ax.set_facecolor('#F0F0F0')
#     ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['black'], zorder=1)
    
#     # Coordenadas do Centro G
#     x_G = (9 * W2) / 2
#     y_G = (2 * H) / 2
    
#     # Plotagem dos Vetores (Escalonados para visibilidade)
#     # Ajuste o fator_escala se as setas ficarem muito pequenas ou grandes
#     # 1. Definir o tamanho máximo visual da seta (ex: 25% da altura do gráfico)
#     tamanho_max_seta = 0.25 * (2 * H) 
    
#     # 2. Encontrar a maior força para servir de referência de escala
#     maior_forca = max(abs(L), abs(D))
    
#     # 3. Calcular o fator de escala dinâmico
#     # O scale do quiver funciona assim: (Valor da Força) / (Escala) = Tamanho no gráfico
#     # Logo, Escala = Maior Força / Tamanho Máximo Desejado
#     escala_dinamica = maior_forca / tamanho_max_seta
    
#     # Vetor Sustentação (Verde)
#     ax.quiver(x_G, y_G, 0, L, color='green', angles='xy', scale_units='xy', 
#               scale=escala_dinamica, width=0.015, label='Sustentação (L)', zorder=5)
    
#     # Vetor Arrasto (Vermelho)
#     ax.quiver(x_G, y_G, D, 0, color='red', angles='xy', scale_units='xy', 
#               scale=escala_dinamica, width=0.015, label='Arrasto (D)', zorder=5)

#     # 5. Caixa de Texto com os Coeficientes (Estilo 'Legend')
#     texto_stats = (f'L: {L:.2f} N\n'
#                    f'D: {D:.2f} N\n'
#                    f'CL: {CL:.4f}\n'
#                    f'CD: {CD:.4f}')
    
#     # Posiciona a caixa no canto superior esquerdo
#     props = dict(boxstyle='round', facecolor='white', alpha=0.8)
#     ax.text(0.05, 0.95, texto_stats, transform=ax.transAxes, fontsize=12,
#             verticalalignment='top', bbox=props, family='monospace')

#     # Detalhes de formatação
#     ax.set_title(f"Diagrama de Forças Aerodinâmicas - $\\theta = {90+theta}^\\circ$ $U_\infty$ = {U_inf} m/s)")
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.axis('equal')
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.legend(loc='lower right')

#     # 6. Lógica de Salvamento
#     diretorio_saida = "resultados2"
#     if not os.path.exists(diretorio_saida):
#         os.makedirs(diretorio_saida)
    
#     nome_figura = f"{diretorio_saida}/forcas_asa-vel{U_inf}_{90+theta}deg.png"
#     fig.savefig(nome_figura, dpi=600, bbox_inches='tight')
#     print(f"Gráfico de forças salvo: {nome_figura}")
    
#     plt.close(fig)

#     return L, D, CL, CD
def plot_comparativo_pressao(dados_dict):
    """
    Gera uma figura comparando a pressão dos dois casos (90° e 15°).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, (label, data) in zip(axes, dados_dict.items()):
        p = data['p']
        m = data['mask']
        p_min = data['min']
        
        # Mascarar para o plot
        p_plot = np.ma.masked_where(m, p)
        
        # Plot de contorno
        niveis = np.linspace(-5000, 600, 100)
        cp = ax.contourf(X, Y, p_plot, levels=niveis, cmap='jet', extend='both')
        
        # Destacar o Aerofólio
        ax.contourf(X, Y, m, levels=[0.5, 1], colors=['black'])
        
        # Texto explicitando o valor mínimo (conforme pedido no item a)
        ax.text(0.05, 0.05, f"P_min: {p_min:.2f} Pa", transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), color='blue', fontweight='bold')
        
        ax.set_title(f"Pressão: {label}")
        ax.axis('equal')
        fig.colorbar(cp, ax=ax, label='Pressão [Pa]')

    plt.tight_layout()
    
    # Salvamento
    caminho = "resultados2/comparativo_pressao_conjunto.png"
    plt.savefig(caminho, dpi=600)
    print(f"Gráfico comparativo salvo em: {caminho}")
    plt.close(fig)

def main():
    def ex1():
        
        velocidades = [10, 30, 90]
        thetas = [0, -75]
        for v_inf in velocidades:
            for theta in thetas:
                mask = make_mask(X, Y, max_domain_width / 2, max_domain_height / 2, H, W1, W2, theta=theta)
                psi = make_psi(X, Y, v_inf, theta, mask, gif=True)
                u_x, u_y, V_abs = make_gradientes(psi, mask, v_inf, theta, dy, dx)
                pressao, min_pressao, max_pressao = calc_pressure(u_x, u_y, v_inf, mask, theta)
                calc_forces(pressao, mask, dx, dy, rho, v_inf, theta)
    def ex2():
        # Parâmetros de entrada
        angulos_ajuste = [0, -75]  # Referencial do seu código (0=Vertical, -75=15°)
        velocidades = [10, 30, 90] # Adicionando as variações do item (b)
        
        # Dicionário para armazenar pressões para o plot conjunto (item a)
        # Guardaremos apenas para a velocidade padrão de 30 m/s para comparar os casos
        pressões_comparativo = {}

        for v_inf in velocidades:
            print(f"\n--- Iniciando simulações para U_inf = {v_inf} m/s ---")
            
            for theta in angulos_ajuste:
                print(f"Processando ângulo: {90+theta}°...")
                
                # 1. Gerar Máscara
                mask = make_mask(X, Y, max_domain_width / 2, max_domain_height / 2, H, W1, W2, theta=theta)
                
                # 2. Solver (PSI)
                psi = make_psi(X, Y, v_inf, theta, mask, gif=False) # Gera GIF só na velocidade base
                
                # 3. Gradientes e Velocidades
                u_x, u_y, V_abs = make_gradientes(psi, mask, v_inf, theta, dy, dx)
                
                # 4. Cálculo de Pressão
                # Ajuste sua calc_pressure para retornar também min_p e max_p
                pressao, min_p, max_p = calc_pressure(u_x, u_y, v_inf, mask, theta)
                
                # Guardar dados para o plot conjunto do item (a) se for a velocidade de 30m/s
                if v_inf == 30:
                    nome_caso = "Placa Vertical" if theta == 0 else "Aerofólio 15°"
                    pressões_comparativo[nome_caso] = {
                        'p': pressao, 
                        'min': min_p,
                        'mask': mask
                    }
                
                # 5. Cálculo e Plot de Forças (item b)
                calc_forces(pressao, mask, dx, dy, rho, U_inf, theta)

        # 6. Após todos os loops, gerar o gráfico comparativo (item a)
        plot_comparativo_pressao(pressões_comparativo)
    
    ex2()
    ex1()

if __name__ == "__main__":
    main()