
import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs): #DDIM **kwargs 가 eta(sigma)임
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long()) #beta -> alpha
            at_next = compute_alpha(b, next_t.long()) #a(t-1)
            xt = xs[-1].to('cuda')
            et = model(xt, t) #et = (xt-sqrt(at)x0)/sqrt(1-at) 모델은 그냥 노이즈를 파악하는거고, 노이즈가 샘플 분포를 따라 생성했으므로 정의됨.
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() #equ 9
            x0_preds.append(x0_t.to('cpu'))

            #sigma scale(eta)가 존재하면 그 값을, 존재하지 않으면 0값(랜덤노이즈 0)을 사용
            c1 = ( #sigma
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            ) #eta ==1, 로 한다면 DDPM과 분산이 동일해짐.

            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et #predicted x0 + random noise + direction pointing to xt
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs): #DDPM
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long()) #그냥 a가 아니라 a 바 임.
            atm1 = compute_alpha(betas, next_t.long()) #a(t-1)
            beta_t = 1 - at / atm1 # = 1-at
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output #예측한 노이즈

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e #예측한 노이즈로부터 x0를 추정. (forward식에서 x0로 정리한 식)
            x0_from_e = torch.clamp(x0_from_e, -1, 1) #모든 값이 -1과 1사이로 제한되도록 함. --> 픽셀/데이터 범위 안정화
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = ( 
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at) #ddpm 평균식

            mean = mean_eps
            noise = torch.randn_like(x) #랜덤 노이즈 (샘플 다양화)
            mask = 1 - (t == 0).float() #마지막 스텝에서 노이즈를 없애기 위해 사용 t==0이면 노이즈를 추가 하지 않음. 따라 mask 0
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log() #현재 스텝에서 사용되는 beta의 log == 분산 값. 베타 틸드 대신 로그 값으로 분산을 대체한다. 완전한 ddpm 샘플링과 같지는 않음.
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
